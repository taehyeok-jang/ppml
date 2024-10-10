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

import functools
import os
from typing import Callable
import json

import re
import jax
import jax.numpy as jn
import numpy as np
import tensorflow as tf  # For data augmentation.
import tensorflow_datasets as tfds
from absl import app, flags
from tqdm import tqdm, trange
import pickle
from functools import partial

import objax
from objax.jaxboard import SummaryWriter, Summary
from objax.util import EasyDict
from objax.zoo import convnet, wide_resnet

from dataset import DataSet

from train import MemModule, network

from collections import defaultdict
FLAGS = flags.FLAGS


def main(argv):
    """
    Perform inference of the saved model in order to generate the
    output logits, using a particular set of augmentations.
    """
    del argv
    tf.config.experimental.set_visible_devices([], "GPU")

    def load(arch):
        return MemModule(network(arch), nclass=100 if FLAGS.dataset == 'cifar100' or FLAGS.dataset == 'purchase100' else 10 , # Modified
                         mnist=FLAGS.dataset == 'mnist',
                         arch=arch,
                         lr=.1,
                         batch=0,
                         epochs=0,
                         weight_decay=0)        

    def cache_load(arch):
        thing = []
        def fn():
            if len(thing) == 0:
                thing.append(load(arch))
            return thing[0]
        return fn

    xs_all = np.load(os.path.join(FLAGS.logdir,"x_train.npy"))[:FLAGS.dataset_size]
    ys_all = np.load(os.path.join(FLAGS.logdir,"y_train.npy"))[:FLAGS.dataset_size]
    
    
    def get_loss(model, xbatch, ybatch, shift, reflect=True, stride=1):
        ############################ Added
        if FLAGS.dataset == "purchase100":
            return np.array(model.model(xbatch, training=True))[:,None,:].repeat(2, axis=1) # shape batch_size, 2, num_out
        ############################ Added
        else:
            outs = []
            for aug in [xbatch, xbatch[:,:,::-1,:]][:reflect+1]:
                aug_pad = tf.pad(aug, [[0] * 2, [shift] * 2, [shift] * 2, [0] * 2], mode='REFLECT').numpy()
                for dx in range(0, 2*shift+1, stride):
                    for dy in range(0, 2*shift+1, stride):
                        this_x = aug_pad[:, dx:dx+32, dy:dy+32, :].transpose((0,3,1,2))

                        logits = model.model(this_x, training=True)
                        outs.append(logits)

            print(np.array(outs).shape)
            return np.array(outs).transpose((1, 0, 2))

    N = FLAGS.bs

    ############################ Added
    if FLAGS.aug == 0: 
        shift = 0
        reflect = False
        stride = 1
    elif FLAGS.aug == 2:
        shift = 0
        reflect = True
        stride = 1
    elif FLAGS.aug == 18:
        shift = 1
        reflect = True
        stride = 1
    elif FLAGS.aug == 50:
        shift = 2
        reflect = True
        stride = 1

    nb_steps = len(range(0, 2*shift+1, stride))
    nb_augmentations = nb_steps**2 * 2 if reflect else nb_steps**2

    if FLAGS.dataset == "purchase100":
        nb_augmentations = 2 # to have standard shapes, no augmentations
    ############################ END Added

    def features(model, xbatch, ybatch):
        return get_loss(model, xbatch, ybatch,
                        shift=shift, reflect=reflect, stride=stride) # Modified
    
    for path in sorted(os.listdir(os.path.join(FLAGS.logdir))):
        if re.search(FLAGS.regex, path) is None:
            print("Skipping from regex")
            continue

        hparams = json.load(open(os.path.join(FLAGS.logdir, path, "hparams.json")))
        arch = hparams['arch']
        model = cache_load(arch)()
        
        logdir = os.path.join(FLAGS.logdir, path)
        
        checkpoint = objax.io.Checkpoint(logdir, keep_ckpts=10, makedir=True)
        max_epoch, last_ckpt = checkpoint.restore(model.vars())
        if max_epoch == 0: continue

        if not os.path.exists(os.path.join(FLAGS.logdir, path, "logits")):
            os.mkdir(os.path.join(FLAGS.logdir, path, "logits"))
        if FLAGS.from_epoch is not None:
            first = FLAGS.from_epoch
        else:
            first = max_epoch-1
            
        for epoch in range(first,max_epoch+1):
            if not os.path.exists(os.path.join(FLAGS.logdir, path, "ckpt", "%010d.npz"%epoch)):
                # no checkpoint saved here
                continue

            ############################ Added
            if os.path.exists(os.path.join(FLAGS.logdir, path, "logits", "%010d_%04d.npy"%(epoch, nb_augmentations))):
                a = np.load(os.path.join(FLAGS.logdir, path, "logits", "%010d_%04d.npy"%(epoch, nb_augmentations)), allow_pickle=True)
                if a.shape[2] == nb_augmentations:
                    print("Skipping already generated file", path, epoch)
                    continue
            ############################ END Added

            try:
                start_epoch, last_ckpt = checkpoint.restore(model.vars(), epoch)
            except:
                print("Fail to load", epoch)
                continue
                
            stats = []

            for i in range(0,len(xs_all),N):
                stats.extend(features(model, xs_all[i:i+N],
                                      ys_all[i:i+N]))
            # This will be shape N, augs, nclass

            np.save(os.path.join(FLAGS.logdir, path, "logits", "%010d_%04d"%(epoch, nb_augmentations)), np.array(stats)[:,None,:,:]) # Modified

if __name__ == '__main__':
    flags.DEFINE_string('dataset', 'cifar10', 'Dataset.')
    flags.DEFINE_string('logdir', 'experiments/', 'Directory where to save checkpoints and tensorboard data.')
    flags.DEFINE_string('regex', '.*experiment.*', 'keep files when matching')
    flags.DEFINE_bool('random_labels', False, 'use random labels.')
    flags.DEFINE_integer('dataset_size', 50000, 'size of dataset.')
    flags.DEFINE_integer('from_epoch', None, 'which epoch to load from.')
    flags.DEFINE_integer('seed_mod', None, 'keep mod seed.')
    flags.DEFINE_integer('modulus', 8, 'modulus.')

    flags.DEFINE_integer('aug', 0, 'number of queries/augmentations') # Added
    flags.DEFINE_integer('bs', 5000, 'Inferring Batch Size.') # Added

    app.run(main)