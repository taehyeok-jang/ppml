import argparse
import logging
import os
import time
from pathlib import Path

from signals import load_input_logits, load_input_labels, convert_signals, load_inputs

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from tqdm import tqdm

from scipy.stats import norm, trim_mean
from sklearn.metrics import auc, roc_curve
from plot import plot_roc

from memory_profiler import profile

# from sklearn.calibration import CalibratedClassifierCV
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split

torch.backends.cudnn.benchmark = True

def get_bins_center(ymedian):
    """
    From all the medians of each bins of equal size between 0 and 1, get the bins center for each y in ymedian.
    """
    length = len(ymedian)
    bins = np.linspace(0, 1, length+1)
    bin_centers = []
    for i in range(1, length+1):
        # Define the bin boundaries
        bin_start = bins[i - 1]
        bin_end = bins[i]
        # Calculate the middle of the current bin
        bin_center = (bin_start + bin_end) / 2
        bin_centers.append(bin_center)
    return bin_centers

def str2bool(v):
  return str(v).lower() in ("true", "1")

def setup_log(report_dir: str, name: str, save_file: bool) -> logging.Logger:
    """Generate the logger for the current run.
    Args:
        report_dir (str): folder name of the audit
        name (str): Logging file name.
        save_file (bool): Flag about whether to save to file.
    Returns:
        logging.Logger: Logger object for the current run.
    """
    my_logger = logging.getLogger(name)
    my_logger.setLevel(logging.INFO)

    if save_file:
        log_format = logging.Formatter(
            "%(asctime)s %(levelname)-8s %(message)s"
            )
        filename = f"{report_dir}/log_{name}.log"

        if not Path(filename).is_file():
            open(filename, 'w+')

        log_handler = logging.FileHandler(filename, mode="w")
        log_handler.setLevel(logging.INFO)
        log_handler.setFormatter(log_format)
        my_logger.addHandler(log_handler)

    return my_logger

def parse_extra(parser, configs):
    """Using a parser and a base config, modify the base config according to the parser
    for CLI bash experiments. E.g. python main.py --cf demo_relative.raml --audit.signal "softmax_relative"
    Args:
        parser (Parser): Parser with the basic arguments
        configs (dict): Dict that we want to change according to the parser
    Returns:
        configs: modified input config
    """
    for key in configs:
        # Generate arguments for top-level keys
        arg_name = '--{}'.format(key)
        parser.add_argument(arg_name, dest=key, default=None,
                            help='{} parameter'.format(arg_name))
        for subkey in configs[key]:
            # Generate arguments for second-level keys
            arg_name = '--{}.{}'.format(key, subkey)
            parser.add_argument(arg_name, dest=subkey, default=None,
                                help='{} parameter'.format(arg_name))
            if isinstance(configs[key][subkey], dict):
                for subsubkey in configs[key][subkey]:
                    # Generate arguments for eventual third-level keys
                    arg_name = '--{}.{}.{}'.format(key, subkey, subsubkey)
                    parser.add_argument(arg_name, dest=subsubkey, default=None,
                                        help='{} parameter'.format(arg_name))
    # Parse command-line arguments
    args, unknown_args = parser.parse_known_args()
    # Update configuration dictionary with command-line arguments
    if args:
        for key in configs:
            if args.__dict__.get(key) is not None:
                configs[key] = args.__dict__.get(key)
            for subkey in configs[key]:
                if args.__dict__.get(subkey) is not None:
                    configs[key][subkey] = args.__dict__.get(subkey)
                if isinstance(configs[key][subkey], dict):
                    for subsubkey in configs[key][subkey]:
                        arg_name = '{}.{}.{}'.format(key, subkey, subsubkey)
                        if args.__dict__.get(subsubkey) is not None:
                            configs[key][subkey][subsubkey] = args.__dict__.get(subsubkey)
    return configs


def create_or_load_signal_file(path,dataset,num_queries=1, num_augmentations=0):
    if not os.path.isfile(path):
        if int(num_augmentations) > 0 :
            if num_queries > 1:
                signal_storage = np.full((len(dataset), num_augmentations, num_queries), np.nan)
            else:
                signal_storage = np.full((len(dataset), num_augmentations), np.nan)
            np.save(path, signal_storage)
        else:
            if num_queries > 1:
                signal_storage = np.full((len(dataset), 1, num_queries), np.nan)
            else:
                signal_storage = np.full((len(dataset), 1), np.nan)
            np.save(path, signal_storage)
    else:
        signal_storage = np.load(path, allow_pickle=True)
    return signal_storage


signal_folder = "signal"

def collect_signal_lira(models_path,
                        model_index, 
                        dataset_indeces, 
                        log_dir, 
                        signal_name, 
                        config, 
                        nb_refs):
    """
    Function that queries a given model on a given dataset on specific indeces for a given signal name 
    and saves it locally for later computations. A folder corresponds to a unique pair of (set of models, parametrized signal)

    Args:
        models_path (str): all models folder path
        model_index (int): model index in the log_dir
        dataset_indeces (List[int]) : which indices to query
        log_dir (str): log_dir 
        signal_name (str): signal's name, usually of the form '{primary_signal}_{post_processing}'
        config (dict): eventual other parameters relative to the signal
        nb_refs (int): number of models inside models_path
    Returns:
        signals: the computed or loaded signals
    """

    epoch = int(configs["data"]["epoch"])
    dataset = np.arange(int(configs["data"]["dataset_size"]))

    # Create a directory will all the primary signals per model 
    if not os.path.exists(f"{log_dir}/{signal_folder}/"):
        os.makedirs(f"{log_dir}/{signal_folder}/")

    nb_augmentations = int(configs["audit"]["nb_augmentation"])

    path_list = models_path.split('/')
    last_folder = f"_{path_list[-1]}" # ref model id

    def load_or_compute_logits():
        ## Loading or Computing the logits
        primary_signal = "logits" + f"_augmented_{nb_augmentations}" * (config["audit"]["augmentation"] == "augmented") + f"_num_ref_{nb_refs}" + last_folder

        if not os.path.exists(f"{log_dir}/{signal_folder}/{primary_signal}/"):
            os.makedirs(f"{log_dir}/{signal_folder}/{primary_signal}/")

        signal_path = f"{log_dir}/{signal_folder}/{primary_signal}/{model_index:04d}.npy"

        model_logits = torch.from_numpy(create_or_load_signal_file(signal_path, dataset, num_queries=10, num_augmentations=nb_augmentations)) # TODO hard coded number of classes
        targets = load_input_labels(models_path)

        if np.isnan(model_logits[dataset_indeces,:]).sum()>0:
            print("load the model and compute logits for model %d" % model_index)
            model_logits, _ = load_input_logits(models_path, epoch, model=model_index, num_augmentations=nb_augmentations)
            
            # base_dataset_path:str, epoch_number:int, model:int=None, num_augmentations:int=2, num_ref_models:int=254,
            model_logits = model_logits[0,:,:,:]
            np.save(signal_path, model_logits.detach().cpu().numpy())
        else:
            print("loading logits for model %d" % model_index)

        return model_logits, targets
    
    if signal_name == "softmax_relative" :
        extra = {}
        temperature = float(config['audit']['temperature'])
        primary_signal = "softmax" + f"_t_{temperature:.1f}" + f"_augmented_{nb_augmentations}" * (config["audit"]["augmentation"] == "augmented") + f"_num_ref_{nb_refs}" + last_folder

        if not os.path.exists(f"{log_dir}/{signal_folder}/{primary_signal}/"):
            os.makedirs(f"{log_dir}/{signal_folder}/{primary_signal}/")

        signal_path = f"{log_dir}/{signal_folder}/{primary_signal}/{model_index:04d}.npy"

        model_signal = torch.from_numpy(create_or_load_signal_file(signal_path, dataset, num_augmentations=nb_augmentations))

        # check if we already queried the primary signals at the given indeces
        # If not, query it
        if torch.isnan(model_signal[dataset_indeces]).sum()>0:
            model_logits, targets = load_or_compute_logits()

            print("load the model and compute signals for model %d" % model_index)

            for k in range(nb_augmentations):
                model_signal[dataset_indeces, k] = convert_signals(model_logits[dataset_indeces, k, :], targets, "softmax", temp=temperature)

            np.save(signal_path, model_signal)
            return model_signal[dataset_indeces]
        else:
            print("loading signals for model %d" % model_index)
            return model_signal[dataset_indeces]
    
    elif signal_name == "sm_softmax_relative" :
        extra = {}
        temperature = float(config['audit']['temperature'])
        extra["taylor_m"] = config['audit']['taylor_m'] # float
        primary_signal = f"softmax_margin_m_{config['audit']['taylor_m']}" + f"_t_{temperature:.1f}" + f"_augmented_{nb_augmentations}" * (config["audit"]["augmentation"] == "augmented") + f"_num_ref_{nb_refs}" + last_folder

        if not os.path.exists(f"{log_dir}/{signal_folder}/{primary_signal}/"):
            os.makedirs(f"{log_dir}/{signal_folder}/{primary_signal}/")

        signal_path = f"{log_dir}/{signal_folder}/{primary_signal}/{model_index:04d}.npy"

        model_signal = torch.from_numpy(create_or_load_signal_file(signal_path, dataset, num_augmentations=nb_augmentations))

        # check if we already queried the primary signals at the given indeces
        # If not, query it
        if torch.isnan(model_signal[dataset_indeces]).sum()>0:
            model_logits, targets = load_or_compute_logits()

            print("load the model and compute signals for model %d" % model_index)

            for k in range(nb_augmentations):
                model_signal[dataset_indeces, k] = convert_signals(model_logits[dataset_indeces, k, :], targets, "soft-margin", temp=temperature, extra=extra)

            np.save(signal_path, model_signal)
            return model_signal[dataset_indeces]
        else:
            print("loading signals for model %d" % model_index)
            return model_signal[dataset_indeces]
        
    elif signal_name == "taylor_softmax_relative" :
        extra = {}
        extra["taylor_n"] = config['audit']['taylor_n']
        temperature = float(config['audit']['temperature'])
        primary_signal = f"taylor_{config['audit']['taylor_n']}" + f"_t_{temperature:.1f}" + f"_augmented_{nb_augmentations}" * (config["audit"]["augmentation"] == "augmented") + f"_num_ref_{nb_refs}" + last_folder

        if not os.path.exists(f"{log_dir}/{signal_folder}/{primary_signal}/"):
            os.makedirs(f"{log_dir}/{signal_folder}/{primary_signal}/")

        signal_path = f"{log_dir}/{signal_folder}/{primary_signal}/{model_index:04d}.npy"

        model_signal = torch.from_numpy(create_or_load_signal_file(signal_path, dataset, num_augmentations=nb_augmentations))

        # check if we already queried the primary signals at the given indeces
        # If not, query it
        if torch.isnan(model_signal[dataset_indeces]).sum()>0:
            model_logits, targets = load_or_compute_logits()

            print("load the model and compute signals for model %d" % model_index)

            for k in range(nb_augmentations):
                model_signal[dataset_indeces, k] = convert_signals(model_logits[dataset_indeces, k, :], targets, "taylor", temp=temperature, extra=extra)

            np.save(signal_path, model_signal)
            return model_signal[dataset_indeces]
        else:
            print("loading signals for model %d" % model_index)
            return model_signal[dataset_indeces]

    elif signal_name == "sm_taylor_softmax_relative" :
        extra = {}
        extra["taylor_m"] = config['audit']['taylor_m'] # float
        extra["taylor_n"] = config['audit']['taylor_n'] # int
        temperature = float(config['audit']['temperature'])
        primary_signal = f"taylor_n_{config['audit']['taylor_n']}_m_{config['audit']['taylor_m']}" + f"_t_{temperature:.1f}" + f"_augmented_{nb_augmentations}" * (config["audit"]["augmentation"] == "augmented") + f"_num_ref_{nb_refs}" + last_folder

        if not os.path.exists(f"{log_dir}/{signal_folder}/{primary_signal}/"):
            os.makedirs(f"{log_dir}/{signal_folder}/{primary_signal}/")

        signal_path = f"{log_dir}/{signal_folder}/{primary_signal}/{model_index:04d}.npy"

        model_signal = torch.from_numpy(create_or_load_signal_file(signal_path, dataset, num_augmentations=nb_augmentations))

        # check if we already queried the primary signals at the given indeces
        # If not, query it
        if torch.isnan(model_signal[dataset_indeces]).sum()>0:
            model_logits, targets = load_or_compute_logits()

            print("load the model and compute signals for model %d" % model_index)

            for k in range(nb_augmentations):
                print(model_logits.shape)
                model_signal[dataset_indeces, k] = convert_signals(model_logits[dataset_indeces, k, :], targets, "taylor-soft-margin", temp=temperature, extra=extra)

            np.save(signal_path, model_signal)
            return model_signal[dataset_indeces]
        else:
            print("loading signals for model %d" % model_index)
            return model_signal[dataset_indeces]
        
    ## Note: To quickly implement another attack you can use another elif signal_name == "signal_name"

    elif signal_name == "loss_logit_rescaled" or signal_name == "loss_reference" or signal_name == "loss_population" or signal_name == "loss_logit_rescaled_relative_direct":

        primary_signal = "loss" + f"_augmented_{nb_augmentations}" * (config["audit"]["augmentation"] == "augmented") + f"_num_ref_{nb_refs}" + last_folder

        if not os.path.exists(f"{log_dir}/signal/{primary_signal}/"):
            os.makedirs(f"{log_dir}/signal/{primary_signal}/")

        signal_path = f"{log_dir}/signal/{primary_signal}/{model_index:04d}.npy"
        # check if the primary signal has been computed
        model_signal = torch.from_numpy(create_or_load_signal_file(signal_path, dataset, num_augmentations=nb_augmentations))

        # check if we already queried the primary signals at the given indeces
        # If not, query it
        if torch.isnan(model_signal[dataset_indeces]).sum()>0:
            model_logits, targets = load_or_compute_logits()
            for k in range(nb_augmentations):
                model_signal[dataset_indeces, k] = convert_signals(model_logits[dataset_indeces, k, :], targets, "log-logit-scaling", temp=0)
            np.save(signal_path, model_signal)
            return model_signal[dataset_indeces]
        else:
            print("loading signals for model %d" % model_index)
            return model_signal[dataset_indeces]
        
    else:
         raise NotImplementedError(f"signal {signal_name} has not yet been implemented...")

def aggregate_signals(signal_name, 
                      membership, 
                      target_signal, 
                      target_indices, 
                      reference_keep_matrix, 
                      reference_signals, 
                      population_indices,
                      configs, 
                      kernel=None, 
                      return_proba_ratio=False):
    """
    Function that aggregate signals for a given signal name under the lira setting. Returns the list of corresponding preds and answers.

    Args:
        signal_name (str): (signal+attack)'s name, e.g. of the form '{signal}_{attack_name}'
        membership (array(dtype=bool)): membership matrix for a given set of target_signals of shape queried_signals
        target_signal (array): signal matrix for a given set of target_signals of shape queried_signals
        target_indices (array): indices of the columns in 1, len(queried_signals) for which to compute membership (e.g. the first 1000)
        reference_keep_matrix (array(dtype=bool)): shape (nb_ref_models-1) x queried_signals
        reference_signals (array): shape (nb_ref_models-1) x queried_signals x nb augmentations
        population_indices (array of ints or 2D array of bools): indices of the columns in len(queried_signals) for which to compute population signal
        configs (dict): eventual other parameters relative to the signal and attack
        kernel (array): optional kernel matrix of distances between all the points in the dataset. Combined with audit.top_k.
        return_proba_ratio (bool) : to specify if we want to just return proba_ratios for relative attacks.
    Returns:
        prediction: the computed of target signals (preferrably, all membership signals should be of order lower=member)
        answers: the membership of target signals
    """

    # in_signals = [] # used for reference models approach (lira, reference, relative)
    out_signals = []

    print(torch.max(target_indices), torch.min(target_indices))

    out_or_in_size = int(configs["audit"]["num_ref_models"]) // 2 # min(configs["train"]["num_in_models"], configs["train"]["num_out_models"])

    # In/Out Data partitioning taken and readapted from https://github.com/carlini/privacy/blob/better-mi/research/mi_lira_2021/plot.py
    for data_idx in range(reference_signals.shape[1]):
        # in_signals.append(
        #     reference_signals[reference_keep_matrix[:, data_idx], data_idx][:out_or_in_size]
        # ) # selects the models that have data_idx as train
        out_signals.append(
            reference_signals[~reference_keep_matrix[:, data_idx], data_idx][:out_or_in_size]
        ) # selects the models that don't have data_idx as train
    
    # in_signals = torch.stack(in_signals) # shape dataset_size x out_or_in_size x nb_augmentations
    out_signals = torch.stack(out_signals) # shape dataset_size x out_or_in_size x nb_augmentations

    pop_target_signals = target_signal[population_indices] # shape pop x out_or_in_size x nb_augmentations

    nb_augmentations = int(configs["audit"]["nb_augmentation"])
    if signal_name == "softmax_relative" or signal_name == "taylor_softmax_relative" or signal_name == "sm_taylor_softmax_relative" or signal_name == "sm_softmax_relative": # relative attack

        def majority_voting_tensor(tensor, axis): # compute majority voting for a bool tensor along a certain axis 
            return torch.mode(torch.stack(tensor), axis).values * 1.0

        if int(configs["audit"]["top_k"]) != -1 : # selects for each x a random set of population equal to int(configs["audit"]["top_k"])
            top_k = int(configs["audit"]["top_k"])
            target_distances = kernel[target_indices][:,population_indices] # shape target_indices x population_indices
            _, indices = target_distances.topk(top_k, largest=False, dim=1)
        
        # reference_signals of shape nb_models x 50k x nb_augmentations
        # out_signal of shape 50k x out_or_in_size x nb_augmentations

        if str2bool(configs["audit"]["offline"]): # only taking out signals
            ref_signals = out_signals.transpose(0, 1) 
            offline_a = float(configs["audit"]["offline_a"])
        elif str(configs["audit"]["offline"]).replace(".", "").isnumeric():
            if float(configs["audit"]["offline"]) <= 1.0 and 0.0 <= float(configs["audit"]["offline"]):
                
                # float(configs["audit"]["offline"] refers to the ratio of offline models
                tot_nb=in_signals.shape[1] # 127
                in_number_of_models=int(tot_nb*(1-float(configs["audit"]["offline"])))
                out_number_of_models=int(tot_nb*float(configs["audit"]["offline"]))

                ref_signals = (torch.cat((in_signals[:,:in_number_of_models], out_signals[:,:out_number_of_models]), dim=1)).transpose(0, 1)

                print(ref_signals.shape) # should be 1, 50000, 18
                
                print(f"Fraction in models with {in_number_of_models} IN Models and {out_number_of_models} OUT models and {out_signals.shape[1]} models")
            else:
                raise Exception ("ERROR : Online Proportion must be between 0.0 (offline) and 1.0 (online)")
        else: # taking half in and half out
            ref_signals = (torch.cat((in_signals, out_signals), dim=1)).transpose(0, 1) # shape nb_chosen_models x dataset_size x nb_augmentations
        
        if configs["audit"]["augmentation"] == "augmented": # using augmentation
        
            if True: # use all augmentations and population (by default)
                proptocut = float(configs["audit"]["proportiontocut"])

                augmented_gammas = [] # contains for each augmentation the boolean matrix (after thresholding by gamma)

                all_mean_x = trim_mean(ref_signals[:,target_indices,:], proportiontocut=proptocut, axis=0)
                all_mean_z = trim_mean(ref_signals[:,population_indices,:], proportiontocut=proptocut, axis=0)

                for k in tqdm(range(0, nb_augmentations), desc=f"Relative attack for each query..."): # computing attack scores for each augmentation

                    mean_x = all_mean_x[:,k]
                    mean_z = all_mean_z[:,k]

                    if str2bool(configs["audit"]["offline"]):
                        if "ymedians" in configs["audit"].keys():
                            ymedian = [float(num) for num in configs["audit"]["ymedians"].split(',')]
                            bin_centers = get_bins_center(ymedian)
                            ymedian = [ymedian[0]] + ymedian + [ymedian[-1]]
                            bin_centers = [0] + bin_centers + [1]

                            prob_ratio_x = (target_signal[target_indices, k].ravel() / (np.interp(mean_x, bin_centers, ymedian)))
                            prob_ratio_z_rev = 1 / (target_signal[population_indices, k].ravel() / (np.interp(mean_z, bin_centers, ymedian)))
                        else: # P(x) = (1+a)/2 P_OUT + (1-a)/2
                            prob_ratio_x = (target_signal[target_indices, k].ravel() / ((1+offline_a)/2 * mean_x + (1-offline_a) /2))
                            prob_ratio_z_rev = 1 / (target_signal[population_indices, k].ravel() / ((1+offline_a)/2 * mean_z + (1-offline_a) /2))
                    else:
                        prob_ratio_x = (target_signal[target_indices, k].ravel() / (mean_x))
                        prob_ratio_z_rev = 1 / (target_signal[population_indices, k].ravel() / (mean_z)) # the inverse to compute the outer product

                    # shape nb_targets x nb_population
                    score = torch.outer(prob_ratio_x, prob_ratio_z_rev)

                    # TODO see if this equals to score = torch.outer(prob_ratio_x, prob_ratio_z_rev)
                    if int(configs["audit"]["top_k"]) != -1 :
                        score = torch.gather(score, 1, indices) # select the top_k z for each x

                    augmented_gammas.append((score > float(configs["audit"]["gamma"])))

                # performs majority voting along augmented axis
                augmented_test = majority_voting_tensor(augmented_gammas, axis=0)
                del augmented_gammas


                prediction = -np.array(augmented_test.mean(1).reshape(1,len(mean_x)))
                prediction[:, population_indices] = ((prediction[:, population_indices] * len(mean_z)) - 1.0) / (len(mean_z) - 1) # to avoid overlapping cases
                answers = np.array(membership[target_indices], dtype=bool)

        else:

            proptocut = float(configs["audit"]["proportiontocut"])

            if len(ref_signals.shape) == 3:
                mean_x = trim_mean(ref_signals[:,target_indices, 0], proportiontocut=proptocut, axis=0)
                mean_z = trim_mean(ref_signals[:,population_indices, 0], proportiontocut=proptocut, axis=0)

                if str2bool(configs["audit"]["offline"]):
                    if "ymedians" in configs["audit"].keys():
                        ymedian = [float(num) for num in configs["audit"]["ymedians"].split(',')]
                        bin_centers = get_bins_center(ymedian)
                        ymedian = [ymedian[0]] + ymedian + [ymedian[-1]]
                        bin_centers = [0] + bin_centers + [1]

                        prob_ratio_x = (target_signal[target_indices, 0].ravel() / (np.interp(mean_x, bin_centers, ymedian)))
                        prob_ratio_z_rev = 1 / (target_signal[population_indices, 0].ravel() / (np.interp(mean_z, bin_centers, ymedian)))
                    else:
                        prob_ratio_x = (target_signal[target_indices, 0].ravel() / ((1+offline_a)/2 * mean_x + (1-offline_a) /2))
                        prob_ratio_z_rev = 1 / (target_signal[population_indices, 0].ravel() / ((1+offline_a)/2 * mean_z + (1-offline_a) /2)) # the inverse to compute quickly
                else:
                    prob_ratio_x = (target_signal[target_indices, 0].ravel() / (mean_x))
                    prob_ratio_z_rev = 1 / (target_signal[population_indices, 0].ravel() / (mean_z)) # the inverse to compute quickly
            else:
                mean_x = trim_mean(ref_signals[:,target_indices], proportiontocut=proptocut, axis=0)
                mean_z = trim_mean(ref_signals[:,population_indices], proportiontocut=proptocut, axis=0)

                if str2bool(configs["audit"]["offline"]):
                    if "ymedians" in configs["audit"].keys():
                        ymedian = [float(num) for num in configs["audit"]["ymedians"].split(',')]
                        bin_centers = get_bins_center(ymedian)
                        ymedian = [ymedian[0]] + ymedian + [ymedian[-1]]
                        bin_centers = [0] + bin_centers + [1]

                        prob_ratio_x = (target_signal[target_indices].ravel() / (np.interp(mean_x, bin_centers, ymedian)))
                        prob_ratio_z_rev = 1 / (target_signal[population_indices].ravel() / (np.interp(mean_z, bin_centers, ymedian)))
                    else:
                        prob_ratio_x = (target_signal[target_indices].ravel() / ((1+offline_a)/2 * mean_x + (1-offline_a) /2))
                        prob_ratio_z_rev = 1 / (target_signal[population_indices].ravel() / ((1+offline_a)/2 * mean_z + (1-offline_a) /2)) # the inverse to compute quickly
                else:
                    prob_ratio_x = (target_signal[target_indices].ravel() / (mean_x))
                    prob_ratio_z_rev = 1 / (target_signal[population_indices].ravel() / (mean_z)) # the inverse to compute quickly
            
            np.savez(f"{log_dir}/{configs['audit']['report_log']}/p_x_{model_index}", # to vizualize p_OUT vs. p_IN
                mean_x=mean_x, # p(x)
                mean_z=mean_z,
                true_mean_x=ref_signals[:,target_indices].mean(0),
                true_mean_z=ref_signals[:,population_indices].mean(0),
                # mean_x_in=in_signals[target_indices,:].mean(1), # shape dataset_size x out_or_in_size x nb_augmentations
                mean_x_out=out_signals[target_indices,:].mean(1),
                # mean_z_in=in_signals[population_indices,:].mean(1),
                mean_z_out=out_signals[population_indices,:].mean(1),
                proba_ratio_x=prob_ratio_x,
                proba_ratio_z=1/prob_ratio_z_rev
            )
             
            final_scores = torch.outer(prob_ratio_x, prob_ratio_z_rev)
            if int(configs["audit"]["top_k"]) != -1 :
                final_scores = torch.gather(final_scores, 1, indices) # select the top_k z for each x

            signal_gamma = -((final_scores > float(configs["audit"]["gamma"]) )*1.0).mean(1).reshape(1,len(mean_x))
            
            prediction = np.array(signal_gamma)
            prediction[:, population_indices] = ((prediction[:, population_indices] * len(mean_z)) - 1.0) / (len(mean_z) - 1) # to avoid overlapping cases
            answers = np.array(membership[target_indices], dtype=bool)

    ## Note: To quickly implement another attack you can use another elif signal_name == "signal_name"

    elif signal_name == "loss_logit_rescaled": 
        # LiRA Attack from Membership Inference Attacks From First Principles from Carlini et al. https://arxiv.org/abs/2112.03570 
        # Taken and readapted from https://github.com/carlini/privacy/blob/better-mi/research/mi_lira_2021/plot.py
        # in_signals = in_signals.detach().cpu().numpy()[target_indices]
        out_signals = out_signals.detach().cpu().numpy()[target_indices] # target indices x nb models x nb augs

        if configs["audit"]["augmentation"] == "augmented":
            # mean_in = np.median(in_signals, 1)
            mean_out = np.median(out_signals, 1)
            
            fix_variance = str2bool(configs["audit"]["fix_variance"])

            if fix_variance:
                if "allout" in configs["audit"].keys() and str2bool(configs["audit"]["allout"]): # reference points are all outside target's training set
                    std_in = np.std(out_signals) # there are no "in_signals"
                    std_out = np.std(out_signals)
                else:
                    # std_in = np.std(in_signals)
                    std_out = np.std(in_signals)
            else:
                # std_in = np.std(in_signals, 1)
                std_out = np.std(out_signals, 1)
            prediction = []
            answers = []
            sc = target_signal[target_indices]
        else:
            if len(target_signal.shape) == 2:
                sc = target_signal[target_indices, 0].reshape(-1, 1) # 50k x 1, no augmentation
                # in_signals = in_signals[:,:,0]
                out_signals = out_signals[:,:,0]
            else:
                sc = target_signal[target_indices] # 50k x 1

            # mean_in = np.median(in_signals, 1).reshape(-1, 1)
            mean_out = np.median(out_signals, 1).reshape(-1, 1)
            fix_variance = str2bool(configs["audit"]["fix_variance"])
            if fix_variance:
                # std_in = np.std(in_signals).reshape(-1, 1)
                std_out = np.std(out_signals).reshape(-1, 1)
            else:
                # std_in = np.std(in_signals, 1).reshape(-1, 1)
                std_out = np.std(out_signals, 1).reshape(-1, 1)

            prediction = []
            answers = []

        if str2bool(configs["audit"]["offline"]):
            pr_in = 0
        else:
            print(0)
            # pr_in = -norm.logpdf(sc, mean_in, std_in + 1e-30)
        pr_out = -norm.logpdf(sc, mean_out, std_out + 1e-30) # gaussian approximation
        score = pr_in - pr_out

        prediction = np.array(score.mean(1))
        answers = np.array(membership[target_indices].reshape(-1, 1), dtype=bool)
    
    elif signal_name == "loss_reference": 
        # Attack-R with linear interpolation (Ye et al.) https://arxiv.org/pdf/2111.09679.pdf
        # Taken and repadated from https://github.com/yuan74/ml_privacy_meter/blob/2022_enhanced_mia/research/2022_enhanced_mia/plot_attack_via_reference_or_distill.py
        if len(target_signal.shape) == 2:
            sc = target_signal[target_indices,0].reshape(-1, 1) # 50k x 1 , no augmentation
            out_signals = out_signals[target_indices,:,0]
        else:
            sc = target_signal[target_indices] # 50k x 1

        def from_correct_logit_to_loss(array): # convert correct logit to the cross entropy loss
            return np.log((1+np.exp(array))/np.exp(array)) # positive
        
        losses = from_correct_logit_to_loss(out_signals).T.numpy() # shape nb_models x nb_target, ref lossses
        check_losses = from_correct_logit_to_loss(sc).T.numpy() # shape nb_target x 1, target losses
    
        dummy_min = np.zeros((1, len(losses[0]))) # shape 1 x nb_target

        dummy_max = dummy_min + 1000 # shape 1 x nb_target

        dat_reference_or_distill = np.sort(np.concatenate((losses, dummy_max, dummy_min), axis=0), axis=0) # shape nb_models + 2 x nb_target 

        prediction = np.array([])

        discrete_alpha = np.linspace(0, 1, len(dat_reference_or_distill))
        for i in range(len(dat_reference_or_distill[0])):
            losses_i =  dat_reference_or_distill[:, i]

            # Create the interpolator
            pr = np.interp(check_losses[0,i], losses_i, discrete_alpha)
            
            prediction = np.append(prediction, pr)
        answers = np.array(membership[target_indices].reshape(-1, 1), dtype=bool)

    elif signal_name == "loss_population": 
        # Attack-P (Ye et al.) https://arxiv.org/pdf/2111.09679.pdf

        def from_correct_logit_to_loss(array): # convert correct logit to the cross entropy loss
            return np.log((1+np.exp(array))/np.exp(array)) # positive

        if len(target_signal.shape) == 2:
            pop_target_signals = pop_target_signals[:,0] # shape pop 
            sc = target_signal[target_indices,0].reshape(1, -1) # 1 x 50k, no augmentation
        else:
            sc = target_signal[target_indices].reshape(1, -1) # 1 x 50k
        
        prediction = []
        answers = []

        pop_target_signals = from_correct_logit_to_loss(pop_target_signals)
        sc = from_correct_logit_to_loss(sc)

        ref = pop_target_signals.reshape(-1, 1).repeat_interleave(len(sc[0]), axis=1) # to parallelize the computation of cdf

        cdf_out = ((ref < sc )*1.0).mean(0)
        score = cdf_out

        prediction = np.array(score)

        answers = np.array(membership[target_indices].reshape(-1, 1), dtype=bool)


    else:
         raise NotImplementedError(f"agg. of signal {signal_name} has not yet been implemented...")

    return prediction, answers

def metric_results(fpr_list, tpr_list, thresholds):
    fprs = [0.01,0.001,0.0001,0.00001,0.0] # 1%, 0.1%, 0.01%, 0.001%, 0%
    tpr_dict = {}
    thresholds_dict = {}
    acc = np.max(1 - (fpr_list + (1 - tpr_list)) / 2)
    roc_auc = auc(fpr_list, tpr_list)

    for fpr in fprs:
        tpr_dict[fpr] = tpr_list[np.where(fpr_list <= fpr)[0][-1]] # tpr at fpr
        thresholds_dict[fpr] = thresholds[np.where(fpr_list <= fpr)[0][-1]] # corresponding threshold

    return roc_auc, acc, tpr_dict, thresholds_dict

def count_subfolders(folder_path):
    if not os.path.exists(folder_path):
        return -1  # Folder doesn't exist
    
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    return len(subfolders)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cf",
        type=str,
        default="experiments/config_models_online.yaml",
        help="Yaml file which contains the configurations",
    )

    # Load the parameters
    args, unknown = parser.parse_known_args()
    with open(args.cf, "rb") as f:
        configs = yaml.load(f, Loader=yaml.Loader)

    configs = parse_extra(parser, configs) # parsing more stuff

    # Set the random seed, log_dir and inference_game
    torch.manual_seed(configs["run"]["random_seed"])
    np.random.seed(configs["run"]["random_seed"])

    log_dir = configs["run"]["log_dir"]

    # Create folders for saving the logs if they do not exist
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    report_dir = f"{log_dir}/{configs['audit']['report_log']}"
    Path(report_dir).mkdir(parents=True, exist_ok=True)

    # Set up the logger
    logger = setup_log(report_dir, "time_analysis", configs["run"]["time_log"])

    start_time = time.time()

    # Load the dataset
    baseline_time = time.time()

    privacy_game = configs["audit"]["privacy_game"]

    ############################
    # Privacy auditing for a model under lira setting
    ############################

    if privacy_game == "privacy_loss_model":

        target_and_test_idx = np.arange(int(configs["data"]["dataset_size"]))

        target_path = configs["data"]["target_dir"]
        nb_models_target = count_subfolders(target_path)
        reference_path = configs["data"]["reference_dir"]
        nb_models_reference = count_subfolders(reference_path)

        is_extra = ("extra_target_dir" in configs["data"].keys()) and (configs["data"]["extra_target_dir"] is not None)

        if is_extra: # for extra non-members signals HAS TO BE IDENTICAL in terms of number as nb_models_target and nb_models_reference
            extra_target_path = configs["data"]["extra_target_dir"]
            nb_models_extra_target = count_subfolders(extra_target_path)
            extra_reference_path = configs["data"]["extra_reference_dir"]
            nb_models_extra_reference = count_subfolders(extra_reference_path)

            assert nb_models_extra_target == nb_models_target and nb_models_extra_reference == nb_models_reference

        logger.info(
                f"Loading from target dir {nb_models_target} target models and reference dir {nb_models_reference} reference models..."
            )



        epoch = int(configs["data"]["epoch"])

        signals = [] # signals or augmented signals
        original_signals = None # for using augmentations as population instead of population itself
        for idx in range(nb_models_target):
            tmp_signal = collect_signal_lira(models_path=target_path,
                                        model_index=idx,
                                        dataset_indeces=target_and_test_idx,
                                        log_dir=log_dir,
                                        signal_name=configs["audit"]["signal"],
                                        config=configs,
                                        nb_refs=nb_models_target)
            signals.append(tmp_signal)
        signals = torch.stack(signals) # shape nb_models x dataset_size x nb_augmentations
        print("Target signals shape: ", signals.shape)

        if is_extra:
            extra_signals = [] # signals or augmented signals
            for idx in range(nb_models_target):
                tmp_signal = collect_signal_lira(models_path=extra_target_path,
                                            model_index=idx,
                                            dataset_indeces=target_and_test_idx,
                                            log_dir=log_dir,
                                            signal_name=configs["audit"]["signal"],
                                            config=configs,
                                            nb_refs=nb_models_target)
                extra_signals.append(tmp_signal)
            
            extra_signals = torch.stack(extra_signals)
            signals =  torch.cat((signals, extra_signals), dim=1) # shape nb_models x dataset_size + extra_set_size x nb_augmentations
            

        reference_signals_lower = [] # only used when the number of ref models is not 254
        if target_path != reference_path:
            for idx in range(nb_models_reference):
                tmp_signal = collect_signal_lira(models_path=reference_path,
                                            model_index=idx,
                                            dataset_indeces=target_and_test_idx,
                                            log_dir=log_dir,
                                            signal_name=configs["audit"]["signal"],
                                            config=configs, 
                                            nb_refs=nb_models_reference)
                reference_signals_lower.append(tmp_signal)
            reference_signals_lower = torch.stack(reference_signals_lower)

            if is_extra:
                extra_reference_signals_lower = []
            
                for idx in range(nb_models_reference):
                    tmp_signal = collect_signal_lira(models_path=extra_reference_path,
                                                model_index=idx,
                                                dataset_indeces=target_and_test_idx,
                                                log_dir=log_dir,
                                                signal_name=configs["audit"]["signal"],
                                                config=configs, 
                                                nb_refs=nb_models_reference)
                    extra_reference_signals_lower.append(tmp_signal)
                
                extra_reference_signals_lower = torch.stack(extra_reference_signals_lower)
                reference_signals_lower =  torch.cat((reference_signals_lower, extra_reference_signals_lower), dim=1) # shape nb_models x dataset_size + extra_set_size x nb_augmentations
        
            print("Reference signals shape: ", reference_signals_lower.shape)
        
        logger.info(
                "Prepare the signals costs %0.5f seconds",
                time.time() - baseline_time,
            )
        baseline_time = time.time()

        if configs["audit"]["target_idx"] == "all":
            list_target_models = np.arange(nb_models_target)
        elif configs["audit"]["target_idx"] == "ten":
            list_target_models = np.arange(10)
        elif configs["audit"]["target_idx"] == "fifty":
            list_target_models = np.arange(50)
        else:
            list_target_models = np.array([int(configs["audit"]["target_idx"])])

        all_aucs = []
        all_accs = []
        all_onep = []
        all_tenth = []
        all_hundredth = []
        all_thousandths = []
        all_zeros = []


        all_logits, keep_matrix = load_input_logits(target_path, epoch, num_augmentations=2)
        _, targets = load_inputs(target_path)

        if is_extra:
            extra_all_logits, extra_keep_matrix = load_input_logits(extra_target_path, epoch, num_augmentations=2)

            extra_keep_matrix[:] = False  # we assume extra is always non-member

            all_logits = torch.cat((all_logits, extra_all_logits), dim=1)
            keep_matrix = torch.cat((keep_matrix, extra_keep_matrix), dim=1) # shape nb_models x dataset_size + extra_dataset_size 

        keep_matrix_lower = None
        if target_path != reference_path:
            ref_logits, keep_matrix_lower = load_input_logits(reference_path, epoch, num_augmentations=2) # we want all_logits to be from the target_dir for consistent subset selection
            _, targets = load_inputs(reference_path)

            if "allout" in configs["audit"].keys(): # reference points are all outside target's training set
                if str2bool(configs["audit"]["allout"]):
                    keep_matrix_lower[:] = False
                
                if is_extra:
                    extra_all_logits, extra_keep_matrix_lower = load_input_logits(extra_reference_path, epoch, num_augmentations=2)
                    extra_keep_matrix_lower[:] = False  # we assume extra is always non-member
                    keep_matrix_lower = torch.cat((keep_matrix_lower, extra_keep_matrix_lower), dim=1) # shape nb_models x dataset_size + extra_dataset_size


        if "subset" in configs["audit"].keys(): # typicality is computed from how high the correct logits are over multiple reference models 

            average_correct_softmax = []
            
            for idx in range(nb_models_target): # we consider the target_dir to be consistent across all experiments
                model_softmax = convert_signals(all_logits[idx,:targets.shape[0],0,:], targets, 'softmax', temp=1.0, extra=None) # size 50k, get the softmax
                average_correct_softmax.append(model_softmax)
            average_correct_softmax = torch.stack(average_correct_softmax).mean(0).numpy().ravel()

            sorted_indices = np.argsort(average_correct_softmax)

            percent = np.ceil(len(average_correct_softmax) * 0.5).astype(int)
            typical_indices = sorted_indices[-percent:] # top 50%
            atypical_indices = sorted_indices[:percent] # bottom 50%

            percent = np.ceil(len(average_correct_softmax) * 0.3).astype(int)
            not_typical_indices = sorted_indices[:-percent] # bottom 70%
            not_atypical_indices = sorted_indices[percent:] # top 70%

            if configs["audit"]["subset"] == "typical": # typicality related to the average difficulty of the sample
                target_and_test_idx = typical_indices
                
            elif configs["audit"]["subset"] == "atypical":
                target_and_test_idx = atypical_indices
            
            elif configs["audit"]["subset"] == "not typical":
                target_and_test_idx = not_typical_indices
            
            elif configs["audit"]["subset"] == "not atypical":
                target_and_test_idx = not_atypical_indices

            elif configs["audit"]["subset"] == "dp outliers":
                target_and_test_idx = np.load("scripts/exp/idx_to_flip.npy")

            print(f"Subset selected...")
        print(np.sort(target_and_test_idx))
        for model_index in list_target_models:

            results_path = f"{log_dir}/{configs['audit']['report_log']}/attack_stats_{model_index}.npz"
            if True:
            # if not os.path.isfile(results_path): # load saved results for model_index
                
                rest_models = np.setdiff1d(np.arange(nb_models_target), [model_index])
                # print(signals.shape, signals[model_index].shape)
                target_signal = signals[model_index]
                membership = keep_matrix[model_index]

                if is_extra:
                    dataset_size = int(configs["data"]["dataset_size"])
                    members_indices = torch.nonzero(membership[np.arange(dataset_size)]).squeeze() # all member indices
                    target_and_test_idx = torch.cat((members_indices, torch.arange(dataset_size, dataset_size+len(members_indices))), dim=0) # all member indices + same size of OOD target data
                    population_indices = np.setdiff1d(np.arange(2*dataset_size), target_and_test_idx) # normal non-members indices

                    print(target_and_test_idx.shape, population_indices.shape)

                if target_path != reference_path: # when the reference models are NOT in the same folder
                    reference_signals = reference_signals_lower
                    reference_keep_matrix = keep_matrix_lower
                else: # when the reference models are in the same folder
                    reference_signals = signals[rest_models, :]
                    reference_keep_matrix = keep_matrix[rest_models]

                if configs["audit"]["signal"] in ["softmax_relative", "taylor_softmax_relative", "sm_taylor_softmax_relative", "sm_softmax_relative"]: # relative attacks (hardcoded)
                    # matrix of random distances between population
                    if int(configs["audit"]["top_k"]) != -1 :
                        pdistances = torch.rand(membership.shape[0], membership.shape[0])
                    else:
                        pdistances = None

                    prediction, answers = aggregate_signals(
                        configs["audit"]["signal"], 
                        membership, 
                        target_signal, 
                        target_and_test_idx, 
                        reference_keep_matrix,
                        reference_signals, 
                        population_indices,
                        configs,
                        kernel=pdistances
                    )
                    fpr_list, tpr_list, betas = roc_curve(answers.ravel(), (-prediction).ravel())
                    prediction = np.array(prediction)

                    if betas[0] > 1: # corrects artefacts caused by roc_curve
                        betas[0] = 1
                    
                    gamma = float(configs["audit"]["gamma"])

                    # to plot the relative score over all x
                    np.savez(f"{log_dir}/{configs['audit']['report_log']}/scores_{model_index}_gamma_{gamma}",
                        scores=prediction, # relative score ratio
                        answers=answers,
                        gamma=gamma
                    )

                    bins= np.linspace(0,1,50)

                    plt.hist((-prediction).ravel()[answers.ravel()==True], bins=bins, label="Member", alpha=0.3)
                    plt.hist((-prediction).ravel()[answers.ravel()==False], bins=bins, label="Non-Member", alpha=0.3)
                    plt.ylabel("Frequency")
                    plt.xlabel("Score")
                    plt.grid(True)
                    plt.legend()
                    plt.savefig(f"{log_dir}/{configs['audit']['report_log']}/score_distribution_{model_index}.png", bbox_inches='tight')
                    plt.clf()

                    # attack results for this gamma
                    roc_auc, acc, tpr_dict, thresholds_dict = metric_results(fpr_list, tpr_list, betas)
                    thresholds = betas
                
                else: # for attacks other than relative attack
                    prediction, answers = aggregate_signals(
                        configs["audit"]["signal"], 
                        membership, 
                        target_signal, 
                        target_and_test_idx, 
                        reference_keep_matrix,
                        reference_signals, 
                        population_indices,
                        configs,
                        kernel=None
                    )
                    fpr_list, tpr_list, thresholds = roc_curve(answers.ravel(), -prediction.ravel())

                    # Last step: compute the metrics
                    roc_auc, acc, tpr_dict, thresholds_dict = metric_results(fpr_list, tpr_list, thresholds)

            print(f"List of top thresholds for model {model_index}",thresholds_dict)
            # saving the attack's results
            np.savez(f"{log_dir}/{configs['audit']['report_log']}/attack_stats_{model_index}",
                model_idx=list_target_models,
                all_aucs=roc_auc,
                all_accs=acc,
                fpr_list=fpr_list,
                tpr_list=tpr_list,
                tpr_dict=tpr_dict,
                thresholds_dict=thresholds_dict
                )
            
            # note : -prediction is positive, i.e. higher = member, so lower=member for prediction
            # other note: thresholds_dict is in the realm of -prediction (i.e. positive)
            correctly_labelled =  ((-prediction).ravel() >= thresholds_dict[0.01]) == (answers.ravel()) # matrix of correctly labelled samples for FPR=1%

            np.savez(f"{log_dir}/{configs['audit']['report_log']}/attack_predictions_{model_index}",
                fpr_list=fpr_list,
                tpr_list=tpr_list,
                thresholds=thresholds, # thresholds for lower = member
                scores=prediction.ravel(), # reversed to have lower = member
                answers=answers.ravel(),
                correct=correctly_labelled
            )

            plot_roc(
                fpr_list,
                tpr_list,
                roc_auc,
                f"{log_dir}/{configs['audit']['report_log']}/{configs['audit']['signal']}_model_{model_index}_attack.png",
            )

            all_aucs.append(roc_auc)
            all_accs.append(acc)
            all_onep.append(tpr_dict[0.01])
            all_tenth.append(tpr_dict[0.001])
            all_hundredth.append(tpr_dict[0.0001])
            all_thousandths.append(tpr_dict[0.00001])
            all_zeros.append(tpr_dict[0.0])

        if configs["audit"]["target_idx"] == "ten":
            logger.info(
                f"Average attack results:"
            )

        elif configs["audit"]["target_idx"] == "all":
                
            np.savez(f"{log_dir}/{configs['audit']['report_log']}/attack_stats",
                model_idx=list_target_models,
                all_aucs=all_aucs,
                all_accs=all_accs,
                all_onep=all_onep,
                all_tenth=all_tenth,
                all_hundredth=all_hundredth,
                all_thousandths=all_thousandths,
                all_zeros=all_zeros
            )
    
            logger.info(
                f"Average attack results:"
            )

        logger.info(
            "AUC %.4f, Accuracy %.4f" % (np.mean(all_aucs), np.mean(all_accs))
        )
        logger.info(
            "TPR@1%%FPR of %.4f, TPR@0.1%%FPR of %.4f, TPR@0.01%%FPR of %.4f, TPR@0.001%%FPR of %.4f, TPR@0%%FPR of %.4f" % (
            np.mean(all_onep), np.mean(all_tenth), np.mean(all_hundredth), np.mean(all_thousandths), np.mean(all_zeros))
        )

    ############################
    # END
    ############################
    logger.info(
        "Run the attack for the all steps costs %0.5f seconds",
        time.time() - start_time,
    )
