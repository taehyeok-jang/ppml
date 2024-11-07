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

parser = argparse.ArgumentParser()
parser.add_argument("--n_queries", default=2, type=int)
parser.add_argument("--model", default="vgg19", type=str)
parser.add_argument("--savedir", default="exp/imagenet-1k", type=str)
args = parser.parse_args()


@torch.no_grad()
def run():
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")

    print("hyper-parameters' settings:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    # Dataset: imagenet-1k
    DATA_DIR = '/serenity/scratch/psml/repo/psml/data/ILSVRC2012'

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    imagenet = datasets.ImageNet(root=DATA_DIR, split='val', transform=transform)
    train_ds, test_ds = random_split(imagenet, [0.8, 0.2])
    
    train_dl = DataLoader(train_ds, batch_size=256, shuffle=False, num_workers=4)

    # Infer the logits with multiple queries
    for path in os.listdir(args.savedir):

        print(f"Loading {args.savedir}/{path}...")
        m = network(args.model)
        m.load_state_dict(torch.load(os.path.join(args.savedir, path, "model.pt"), weights_only=True))
        
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

        np.save(os.path.join(args.savedir, path, "logits.npy"), logits_n)

    
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
