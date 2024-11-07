# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified copy by Chenxiang Zhang (orientino) of the original:
# https://github.com/tensorflow/privacy/tree/master/research/mi_lira_2021


import argparse
import functools
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from sklearn.metrics import auc, roc_curve

from datetime import datetime

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

parser = argparse.ArgumentParser()
parser.add_argument("--savedir", default="exp/cifar10", type=str)
args = parser.parse_args()


'''
## DATASET SPLIT STRATEGY 

imagenet-1k: 

- 'train': T1 + T2 + T3(others)
    |T1| = 40K
    |T2| = 10K
    |T3| = 1.28M - 40K - 10K 

- 'val': S1 + S2 
    |S1| = 40K
    |S2| = 10K

- 'test': not used 


step 1: train.py => train shadow models;

    - for training shadow i; 
        train set: (T1+S1)_i
        eval set: (T1'+S1')
        
step 2: inference.py => 각 shadow models 별로 logits 출력하여 저장함.

step 3: score.py => 그냥 logit scaling

step 4: plot.py (eval) => membership inference attack to victim model. 

    that is, evalulate per data point (x, y).
    
    for (x_1, y_1) ... (x_k, y_k):
    
        for 1 ... N shadow models: 
        => 
        (x_i, y_i) -> N/2 of IN shadow models -> Q_in
        (x_i, y_i) -> N/2 of OUT shadow models -> Q_out
        
        f_ = query to victim model.
        
        final_score = p(f_|Q_in)/p(f_|Q_out). => by 
           
    
    

4-0. 
    load imagenet-1k 'train', and split by T1 + T2 + T3 (T1, T2, T3 must be always same indices across systems)
    load imagenet-1k 'val',   and split by S1 + S2      (S1, S2 must be always same indices) 



''' 

'''
- score: 
φ(f(x)y). (logit scaling of p. log(p/(1-p), p=f(x)y) 

>>> score 
array([[10.13168889,  4.40828625],
       [ 9.08919234, 14.36410262],
       [12.61739964, 11.21139374],
       ...,
       [10.15480552,  8.78174738],
       [ 7.48064894,  9.25800511],
       [ 8.50198769,  7.83281414]])
>>> score.shape
(50000, 2)

- keep: 
ground-truth labels of membership. 

if true, 

>>> keep
array([False, False, False, ...,  True, False, False])

>>> keep.shape
(50000,)
'''
def sweep(score, x):
    """
    Compute a ROC curve and then return the FPR, TPR, AUC, and ACC.
    """
    fpr, tpr, _ = roc_curve(x, -score)
    acc = np.max(1 - (fpr + (1 - tpr)) / 2)
    return fpr, tpr, auc(fpr, tpr), acc


def load_data():
    """
    Load our saved scores and then put them into a big matrix.
    """
    global scores, keep
    scores = []
    keep = []

    for path in os.listdir(args.savedir):
        scores.append(np.load(os.path.join(args.savedir, path, "scores.npy")))
        keep.append(np.load(os.path.join(args.savedir, path, "keep.npy")))
    scores = np.array(scores)
    keep = np.array(keep)

    return scores, keep


def generate_ours(keep, scores, check_keep, check_scores, in_size=100000, out_size=100000, fix_variance=False):
    """
    Fit a two predictive models using keep and scores in order to predict
    if the examples in check_scores were training data or not, using the
    ground truth answer from check_keep.
    """
    dat_in = []
    dat_out = []

    # scores.shape[1] = 40K
    for j in range(scores.shape[1]):
        dat_in.append(scores[keep[:, j], j, :])
        dat_out.append(scores[~keep[:, j], j, :])

    in_size = min(min(map(len, dat_in)), in_size)
    out_size = min(min(map(len, dat_out)), out_size)

    dat_in = np.array([x[:in_size] for x in dat_in])    # dat_in:  (40K, n_shadows-1, 2)
    dat_out = np.array([x[:out_size] for x in dat_out]) # dat_out:  (40K, n_shadows-1, 2)

    mean_in = np.median(dat_in, 1)
    mean_out = np.median(dat_out, 1)

    if fix_variance:
        std_in = np.std(dat_in)
        std_out = np.std(dat_in)
    else:
        std_in = np.std(dat_in, 1)
        std_out = np.std(dat_out, 1)

    prediction = []
    answers = []
    
    # check_keep/scores are used as data distribution of a target model, selected from one of trained models.
    # check_keep:  (1, 50000)
    # check_scores:  (1, 50000, 2)
    
    for ans, sc in zip(check_keep, check_scores):
        pr_in = -scipy.stats.norm.logpdf(sc, mean_in, std_in + 1e-30)
        pr_out = -scipy.stats.norm.logpdf(sc, mean_out, std_out + 1e-30)
        score = pr_in - pr_out

        prediction.extend(score.mean(1))
        answers.extend(ans)

    return prediction, answers


def generate_ours_offline(keep, scores, check_keep, check_scores, in_size=100000, out_size=100000, fix_variance=False):
    """
    Fit a single predictive model using keep and scores in order to predict
    if the examples in check_scores were training data or not, using the
    ground truth answer from check_keep.
    """
    dat_in = []
    dat_out = []

    for j in range(scores.shape[1]):
        dat_in.append(scores[keep[:, j], j, :])
        dat_out.append(scores[~keep[:, j], j, :])

    out_size = min(min(map(len, dat_out)), out_size)

    dat_out = np.array([x[:out_size] for x in dat_out])

    mean_out = np.median(dat_out, 1)

    if fix_variance:
        std_out = np.std(dat_out)
    else:
        std_out = np.std(dat_out, 1)

    prediction = []
    answers = []
    for ans, sc in zip(check_keep, check_scores):
        score = scipy.stats.norm.logpdf(sc, mean_out, std_out + 1e-30)

        prediction.extend(score.mean(1))
        answers.extend(ans)
    return prediction, answers


def generate_global(keep, scores, check_keep, check_scores):
    """
    Use a simple global threshold sweep to predict if the examples in
    check_scores were training data or not, using the ground truth answer from
    check_keep.
    """
    prediction = []
    answers = []
    for ans, sc in zip(check_keep, check_scores):
        prediction.extend(-sc.mean(1))
        answers.extend(ans)

    return prediction, answers


def do_plot(fn, keep, scores, ntest, legend="", metric="auc", sweep_fn=sweep, **plot_kwargs):
    """
    Generate the ROC curves by using ntest models as test models and the rest to train.
    """

    prediction, answers = fn(keep[:-ntest], scores[:-ntest], keep[-ntest:], scores[-ntest:])

    fpr, tpr, auc, acc = sweep_fn(np.array(prediction), np.array(answers, dtype=bool))

    low = tpr[np.where(fpr < 0.001)[0][-1]]

    print("Attack %s   AUC %.4f, Accuracy %.4f, TPR@0.1%%FPR of %.4f" % (legend, auc, acc, low))

    metric_text = ""
    if metric == "auc":
        metric_text = "auc=%.3f" % auc
    elif metric == "acc":
        metric_text = "acc=%.3f" % acc

    plt.plot(fpr, tpr, label=legend + metric_text, **plot_kwargs)
    return (acc, auc)


def fig_fpr_tpr():
    plt.figure(figsize=(4, 3))

    do_plot(generate_ours, keep, scores, 1, "Ours (online)\n", metric="auc")

    do_plot(functools.partial(generate_ours, fix_variance=True), keep, scores, 1, "Ours (online, fixed variance)\n", metric="auc")

    do_plot(functools.partial(generate_ours_offline), keep, scores, 1, "Ours (offline)\n", metric="auc")

    do_plot(functools.partial(generate_ours_offline, fix_variance=True), keep, scores, 1, "Ours (offline, fixed variance)\n", metric="auc")

    do_plot(generate_global, keep, scores, 1, "Global threshold\n", metric="auc")

    plt.semilogx()
    plt.semilogy()
    plt.xlim(1e-5, 1)
    plt.ylim(1e-5, 1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.plot([0, 1], [0, 1], ls="--", color="gray")
    plt.subplots_adjust(bottom=0.18, left=0.18, top=0.96, right=0.96)
    plt.legend(fontsize=8)
    
    current = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"figures/fprtpr_{current}.png")
    
#     plt.savefig("fprtpr.png")
    plt.show()


if __name__ == "__main__":
    load_data()
    fig_fpr_tpr()
