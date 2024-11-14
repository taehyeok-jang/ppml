3# Copyright 2021 Google LLC
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
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10, CIFAR100

import pytorch_lightning as pl

parser = argparse.ArgumentParser()
parser.add_argument("--savedir", default="exp/cifar10", type=str)
parser.add_argument("--dataset", default="", type=str)
parser.add_argument("--mode", default="", type=str) # train / eval

args = parser.parse_args()

seed = 1583745484

def load_one(path):

    print(f"processing {path}...")
    
    pl.seed_everything(seed)
    
    """
    This loads a logits and converts it to a scored prediction.
    """
    if args.mode == "train":
        print("use logits.npy") 
        opredictions = np.load(os.path.join(path, "logits.npy")) # [n_examples, n_augs, n_classes]
    elif args.mode == "eval":
        print("use logits_eval.npy")
        opredictions = np.load(os.path.join(path, "logits_eval.npy")) # [n_examples, n_augs, n_classes]
    else:
        raise ValueError("unknown mode")
    
    # Be exceptionally careful.
    # Numerically stable everything, as described in the paper.
    predictions = opredictions - np.max(opredictions, axis=-1, keepdims=True)
    predictions = np.array(np.exp(predictions), dtype=np.float64)
    predictions = predictions / np.sum(predictions, axis=-1, keepdims=True)

    labels = get_labels()  # TODO generalize this

    COUNT = predictions.shape[0]
    y_true = predictions[np.arange(COUNT), :, labels[:COUNT]]

    print("mean acc", np.mean(predictions[:, 0, :].argmax(1) == labels[:COUNT]))

    predictions[np.arange(COUNT), :, labels[:COUNT]] = 0
    y_wrong = np.sum(predictions, axis=-1)

    logit = np.log(y_true + 1e-45) - np.log(y_wrong + 1e-45)
    np.save(os.path.join(path, "scores.npy"), logit)


def get_labels():
    torch.manual_seed(seed)
    datadir = Path().home() / "dataset"

    if args.dataset == "cifar10":
        print("import cifar10...")
        train_ds = CIFAR10(root=datadir, train=True, download=True)
    elif args.dataset == "cifar100":
        print("import cifar100...")
        train_ds = CIFAR100(root=datadir, train=True, download=True)
    else:
        raise ValueError("undefined dataset")
    
    train_ds, eval_ds = random_split(train_ds, [0.8, 0.2])

    # print("train_ds: ", len(train_ds))
    # print(train_ds.indices[:100])
    # print("eval_ds: ", len(eval_ds))
    # print(eval_ds.indices[:100])

    if args.mode == "train":
        print("returning train_targets") 
        all_targets = np.array(train_ds.dataset.targets)
        targets = all_targets[train_ds.indices]
    elif args.mode == "eval":
        print("returning eval_targets") 
        all_targets = np.array(eval_ds.dataset.targets)
        targets = all_targets[eval_ds.indices]
    else:
        raise ValueError("unknown mode")
    
    return targets


def load_stats():
    with mp.Pool(8) as p:
        p.map(load_one, [os.path.join(args.savedir, x) for x in os.listdir(args.savedir)])


if __name__ == "__main__":
    load_stats()
