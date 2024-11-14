# PyTorch implementation of
# https://github.com/tensorflow/privacy/blob/master/research/mi_lira_2021/train.py
#
# author: Chenxiang Zhang (orientino)

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import models, transforms
import timm
from torchvision.datasets import CIFAR10, CIFAR100
from tqdm import tqdm

from wide_resnet import WideResNet

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="resnet18", type=str)
parser.add_argument("--dataset", default="", type=str)
parser.add_argument("--epochs", default=1, type=int)
parser.add_argument("--lr", default=0.1, type=float)
parser.add_argument("--n_shadows", default=16, type=int)
parser.add_argument("--shadow_id", default=1, type=int)
parser.add_argument("--pkeep", default=0.5, type=float)
parser.add_argument("--savedir", default="exp/cifar10", type=str)
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")

# seed = np.random.randint(0, 1000000000)
# seed ^= int(time.time())
seed = 1583745484

def run():

    pl.seed_everything(seed)

    args.debug = True
    wandb.init(project="lira", mode="disabled" if args.debug else "online")
    wandb.config.update(args)
    
    print("parameter settings:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    # Dataset

    if args.model == "vit_large_patch16_224":
        train_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        test_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else: 
        train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
            ]
        )

    
    torch.manual_seed(seed)
    datadir = Path().home() / "dataset"

    if args.dataset == "cifar10":
        print("import cifar10...")
        train_ds = CIFAR10(root=datadir, train=True, download=True, transform=train_transform)
        test_ds = CIFAR10(root=datadir, train=False, download=True, transform=test_transform)
    elif args.dataset == "cifar100":
        print("import cifar100...")
        train_ds = CIFAR100(root=datadir, train=True, download=True, transform=train_transform)
        test_ds = CIFAR100(root=datadir, train=False, download=True, transform=test_transform)
    else:
        raise ValueError("undefined dataset")


    train_ds, eval_ds = random_split(train_ds, [0.8, 0.2])

    print("train_ds: ", len(train_ds))
    print(train_ds.indices[:100])
    print("eval_ds: ", len(eval_ds))
    print(eval_ds.indices[:100])

    # Compute the IN / OUT subset:
    # If we run each experiment independently then even after a lot of trials
    # there will still probably be some examples that were always included
    # or always excluded. So instead, with experiment IDs, we guarantee that
    # after `args.n_shadows` are done, each example is seen exactly half
    # of the time in train, and half of the time not in train.

    size = len(train_ds)
    np.random.seed(seed)
    if args.n_shadows is not None:
        np.random.seed(0)
        keep = np.random.uniform(0, 1, size=(args.n_shadows, size))
        order = keep.argsort(0)
        keep = order < int(args.pkeep * args.n_shadows)
        keep = np.array(keep[args.shadow_id], dtype=bool)
        keep = keep.nonzero()[0]
    else:
        keep = np.random.choice(size, size=int(args.pkeep * size), replace=False)
        keep.sort()
    keep_bool = np.full((size), False)
    keep_bool[keep] = True

    train_ds = torch.utils.data.Subset(train_ds, keep)
    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4)
    test_dl = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=4)

    # Model
    m = network(args.model, pretrained_=True)
    m = m.to(DEVICE)
    print(m)
    # For efficient fine-tune, freeze some intermediate layers within model
    m = freeze_interdemidate_layers(m, args.model)
    
    optim = torch.optim.SGD(m.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    # Train
    for epoch in range(args.epochs):
        m.train()
        loss_total = 0
        pbar = tqdm(train_dl)
        for itr, (x, y) in enumerate(pbar):
            x, y = x.to(DEVICE), y.to(DEVICE)

            outputs = m(x)
            loss = F.cross_entropy(outputs, y)
            loss_total += loss
            
#             print("x: ", x.shape, x)
#             print("y: ", y.shape, y)
#             print("outputs: ", outputs.shape, outputs)

            pbar.set_postfix_str(f"loss: {loss:.2f}")
            optim.zero_grad()
            loss.backward()
            optim.step()
        sched.step()

        test_acc = get_acc(m, test_dl)
        print(f"[Epoch {epoch}] Test Accuracy: {test_acc:.4f}")
        wandb.log({"loss": loss_total / len(train_dl)})

    print(f"[test] acc_test: {get_acc(m, test_dl):.4f}")
    wandb.log({"acc_test": get_acc(m, test_dl)})

    savedir = os.path.join(args.savedir, str(args.shadow_id))
    os.makedirs(savedir, exist_ok=True)
    np.save(savedir + "/keep.npy", keep_bool)
    torch.save(m.state_dict(), savedir + "/model.pt")
    
    
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
        raise ValueError(f"Model {arch} not available.")

    dataset_classes = {"cifar10": 10, "cifar100": 100}
    n_classes = dataset_classes.get(args.dataset)
    
    if not n_classes:
        raise ValueError(f"Unsupported dataset '{args.dataset}'")

    if args.model == "vgg19": # for VGG-19
        num_features = model.classifier[6].in_features
        m.classifier[6] = nn.Linear(num_features, n_classes)      
    elif args.model == "vit_large_patch16_224": # for ViT (vision transformer) 
        num_features = model.head.in_features
        model.head = nn.Linear(num_features, n_classes)        
    elif args.model == "efficientnet_b7": # for efficientnet
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 100)
    else:
        raise ValueError("undefined dataset")    
        
    return model

    
def freeze_interdemidate_layers(model, model_name):
    """
    Freeze specific layers of a model for efficient fine-tuning.
    
    Parameters:
    - model (torch.nn.Module): The model to apply freezing to.
    - model_name (str): The name of the model.
    """
    #if model_name == "vgg19":
    #    print("Freezing VGG-19 intermediate layers...")
    #    for param in model.features.parameters():
    #        param.requires_grad = False
    
    if model_name == "vit_large_patch16_224":
        print("Freezing ViT-Large intermediate layers...")
        for name, param in model.named_parameters():
            if "head" not in name:
                param.requires_grad = False
    
    #elif model_name == "efficientnet_b7":
    #    print("Freezing EfficientNet-B7 intermediate layers...")
    #    for param in model.parameters():
    #        param.requires_grad = False
    #    for param in model.classifier[1].parameters():
    #        param.requires_grad = True
    
    else:
        print(f"Do not freeze layers for model: {model_name}")
    
    return model
        
@torch.no_grad()
def get_acc(model, dl):
    acc = []
    for x, y in dl:
        x, y = x.to(DEVICE), y.to(DEVICE)
        acc.append(torch.argmax(model(x), dim=1) == y)
    acc = torch.cat(acc)
    acc = torch.sum(acc) / len(acc)

    return acc.item()


if __name__ == "__main__":
    run()
