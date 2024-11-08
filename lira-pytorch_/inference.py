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
from torchvision.datasets import CIFAR10
from tqdm import tqdm

from wide_resnet import WideResNet

import pytorch_lightning as pl

parser = argparse.ArgumentParser()
parser.add_argument("--n_queries", default=2, type=int)
parser.add_argument("--model", default="vgg19", type=str)
parser.add_argument("--savedir", default="exp/cifar10", type=str)
parser.add_argument("--mode", default="train", type=str)
args = parser.parse_args()


seed = 1583745484

@torch.no_grad()
def run():

    pl.seed_everything(seed)

    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")

    print("parameter settings:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    # Dataset
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
        ]
    )
    
    torch.manual_seed(seed)
    datadir = Path().home() / "dataset"
        
    train_ds = CIFAR10(root=datadir, train=True, download=True, transform=transform)
    train_ds, eval_ds = random_split(train_ds, [0.8, 0.2])

    print("train_ds: ", len(train_ds))
    print(train_ds.indices[:100])
    print("eval_ds: ", len(eval_ds))
    print(eval_ds.indices[:100])

    if args.mode == "train":
        print("use train_ds") 
        dl_ = DataLoader(train_ds, batch_size=128, shuffle=False, num_workers=4)    
    elif args.mode == "eval":
        print("use eval_ds")
        dl_ = DataLoader(eval_ds, batch_size=128, shuffle=False, num_workers=4)    
    else:
        raise ValueError("unknown mode")
    

    print(f"Loading {args.savedir}...")
    m = network(args.model, pretrained_=True)
    m.load_state_dict(torch.load(os.path.join(args.savedir, "model.pt"), weights_only=True))

    # Wrap the model with DataParallel to use multiple GPUs
    # if torch.cuda.device_count() > 1:
    #     print(f"Using {torch.cuda.device_count()} GPUs for parallel processing.")
    #     m = torch.nn.DataParallel(m)

    m.to(DEVICE)
    m.eval()

    logits_n = []
    for i in range(args.n_queries):
        logits = []
        for x, _ in tqdm(dl_):
            x = x.to(DEVICE)
            outputs = m(x)

            print(outputs)
            # if torch.isnan(outputs).any():
            #    print("NaNs detected in outputs")
            #    raise ValueError("NaNs detected in outputs")
            
            logits.append(outputs.cpu().numpy())

        # print("logits: ")
        print(logits)
        logits_n.append(np.concatenate(logits))
        
    logits_n = np.stack(logits_n, axis=1)
    print("logits_n: ", logits_n.shape) 
    # print(logits_n)

    if args.mode == "train":
        print("save to logits.npy") 
        np.save(os.path.join(args.savedir, "logits.npy"), logits_n)
    elif args.mode == "eval":
        print("save to logits_eval.npy") 
        np.save(os.path.join(args.savedir, "logits_eval.npy"), logits_n)
    else:
        raise ValueError("unknown mode")
        
    


def network(arch: str, pretrained_: bool):
    print(f'arch: {arch}, pretrained: {pretrained_}') 
    
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
        model = models.__dict__[arch](pretrained=pretrained_)
    elif arch in PYTORCH_IMAGE_MODELS:
        model = timm.create_model(arch, pretrained=pretrained_)
    else:
        raise ValueError(f"Model {model_name} not available.")
        
    # for VGG-19
    model.classifier[6] = torch.nn.Linear(in_features=4096, out_features=10)
    return model


if __name__ == "__main__":
    run()
