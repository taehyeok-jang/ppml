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
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
from torchvision.datasets import CIFAR10
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl

parser = argparse.ArgumentParser()
parser.add_argument("--savedir", default="exp/imagenet-1k", type=str)
args = parser.parse_args()


def load_one(path):
    # get fixed splits from random_split()
    SEED = 1583745484
    pl.seed_everything(SEED)
    
    """
    This loads a logits and converts it to a scored prediction.
    """
    opredictions = np.load(os.path.join(path, "logits.npy"))  # [n_examples, n_augs, n_classes]
    
    print(f"processing {path}...")
    

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
    DATA_DIR = '/serenity/scratch/psml/repo/psml/data/ILSVRC2012'

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    imagenet = datasets.ImageNet(root=DATA_DIR, split='val', transform=transform)
    train_ds, test_ds = random_split(imagenet, [0.8, 0.2])
    
    targets_ = [train_ds.dataset.targets[i] for i in train_ds.indices]
    
    return np.array(targets_)


def load_stats():
    with mp.Pool(8) as p:
        p.map(load_one, [os.path.join(args.savedir, path) for path in os.listdir(args.savedir)])


if __name__ == "__main__":
    load_stats()
