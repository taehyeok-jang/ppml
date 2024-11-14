# PyTorch implementation of
# https://github.com/tensorflow/privacy/blob/master/research/mi_lira_2021/inference.py
#
# author: Chenxiang Zhang (orientino)

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import models, transforms
from torchvision import datasets
from torchvision.datasets import CIFAR10
from tqdm import tqdm

from wide_resnet import WideResNet

import pytorch_lightning as pl

parser = argparse.ArgumentParser()
parser.add_argument("--n_queries", default=2, type=int)
parser.add_argument("--model", default="vgg19", type=str)
parser.add_argument("--savedir", default="exp/imagenet-1k", type=str)
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

    for every shadow model, just performance for all data points (T1+S1). 

    논문 보면서 align 좀 해야겠다.. ㅜㅜ

step 3: score.py => 그냥 logit scaling

step 4: plot.py (eval) => membership inference attack to victim model. 
    

4-0. 
    load imagenet-1k 'train', and split by T1 + T2 + T3 (T1, T2, T3 must be always same indices across systems)
    load imagenet-1k 'val',   and split by S1 + S2      (S1, S2 must be always same indices) 



''' 
    
@torch.no_grad()
def run():
    # get fixed splits from random_split() 
    SEED = 1583745484
    pl.seed_everything(SEED)
    
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")

    print("hyper-parameters' settings:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    # Dataset: imagenet-1k
    DATA_DIR = '/serenity/scratch/psml/data/ILSVRC2012'

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
        
    imagenet_t = datasets.ImageNet(root=DATA_DIR, split='train', transform=transform)
    T_train_SIZE = 20000
    T_eval_SIZE = 5000
    T_others_SIZE = len(imagenet_t) - T_train_SIZE - T_eval_SIZE
    T_train, T_eval, _ = random_split(imagenet_t, [T_train_SIZE, T_eval_SIZE, T_others_SIZE])
    
    imagenet_s = datasets.ImageNet(root=DATA_DIR, split='val', transform=transform)
    S_train_SIZE = 20000
    S_eval_SIZE = 5000
    S_others_SIZE = len(imagenet_s) - S_train_SIZE - S_eval_SIZE
    S_train, S_eval, _ = random_split(imagenet_s, [S_train_SIZE, S_eval_SIZE, S_others_SIZE])
    
    print("T_train: ", T_train.indices[:100])
    print("T_eval: ", T_eval.indices[:100])
    print("S_train: ", S_train.indices[:100])
    print("S_eval: ", S_eval.indices[:100])

    train_ds = ConcatDataset([T_train, S_train])
#     test_ds = ConcatDataset([T_eval, S_eval])
    
    train_dl = DataLoader(train_ds, batch_size=256, shuffle=False, num_workers=4)

    # Infer the logits with multiple queries
    print(f"Loading {args.savedir}...")
    m = network(args.model)
    m.load_state_dict(torch.load(os.path.join(args.savedir, "model.pt"), weights_only=True))

    # Wrap the model with DataParallel to use multiple GPUs
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for parallel processing.")
        m = torch.nn.DataParallel(m)

    m.to(DEVICE)
    m.eval()

    logits_n = []
    for i in range(args.n_queries):
        logits = []
        for x, _ in tqdm(train_dl):
            x = x.to(DEVICE)
            outputs = m(x)
            logits.append(outputs.cpu().numpy())
        logits_n.append(np.concatenate(logits))
    logits_n = np.stack(logits_n, axis=1)
    print(logits_n.shape)

    np.save(os.path.join(args.savedir, "logits.npy"), logits_n)

    
def network(arch: str):
    # https://pytorch.org/vision/stable/models.html
    TORCHVISION_MODELS = ['resnet18', 'resnet50', 'resnet101', 'vgg16', 'vgg19', 'densenet121', 
                          'wide_resnet50_2', 'wide_resnet101_2',
                          'densenet201', 'mobilenet_v2', 'inception_v3', 
                          'efficientnet_b0', 'efficientnet_b7', 
                          'squeezenet1_0', 'alexnet', 'googlenet', 'shufflenet_v2_x1_0']
    
    # https://github.com/huggingface/pytorch-image-models
    PYTORCH_IMAGE_MODELS = ['vit_base_patch16_224', 'vit_large_patch16_224', 'deit_base_patch16_224',
                        'convnext_base', 'convnext_large']
    
    if arch in TORCHVISION_MODELS:
        return models.__dict__[arch](pretrained=True)
    elif arch in PYTORCH_IMAGE_MODELS:
        return timm.create_model(arch, pretrained=True)
    else:
        raise ValueError(f"Model {model_name} not available.")

        
if __name__ == "__main__":
    run()
