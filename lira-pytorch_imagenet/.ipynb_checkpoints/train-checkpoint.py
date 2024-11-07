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
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
from torchvision import models, transforms
from torchvision.datasets import CIFAR10
import timm # state-of-the-art models (e.g. vit...)

from tqdm import tqdm

from wide_resnet import WideResNet

parser = argparse.ArgumentParser()
parser.add_argument("--lr", default=0.01, type=float)
parser.add_argument("--weight_decay", default=0.0005, type=float)
parser.add_argument("--epochs", default=100, type=int)
parser.add_argument("--n_shadows", default=16, type=int)
parser.add_argument("--shadow_id", default=1, type=int)
parser.add_argument("--model", default="", type=str)
parser.add_argument("--pkeep", default=0.5, type=float)
parser.add_argument("--savedir", default="exp/imagenet-1k", type=str)
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")


def run():
#     seed = np.random.randint(0, 1000000000)
#     seed ^= int(time.time())
    SEED = 1583745484
    pl.seed_everything(SEED)

    args.debug = True
    wandb.init(project="lira", mode="disabled" if args.debug else "online")
    wandb.config.update(args)
    
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
    
    # Compute the IN / OUT subset:
    # If we run each experiment independently then even after a lot of trials
    # there will still probably be some examples that were always included
    # or always excluded. So instead, with experiment IDs, we guarantee that
    # after `args.n_shadows` are done, each example is seen exactly half
    # of the time in train, and half of the time not in train.

    size = len(train_ds)
    np.random.seed(SEED)
    if args.n_shadows is not None:
        np.random.seed(SEED)
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

    m = network(args.model)
    m = m.to(DEVICE)

    optim = torch.optim.SGD(m.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    # Train
    for epoch in tqdm(range(args.epochs), desc="training epochs..."):
        m.train()
        
        loss_total = 0
        pbar = tqdm(train_dl)
        for itr, (x, y) in enumerate(pbar):
            
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            outputs = m(x)
            loss = F.cross_entropy(outputs, y)
            loss_total += loss.item()

            pbar.set_postfix_str(f"loss: {loss:.2f}")
            optim.zero_grad()
            loss.backward()
            optim.step()
        sched.step()

        wandb.log({"loss": loss_total / len(train_dl), "epoch": epoch})

        test_acc = get_acc(m, test_dl)
        print(f"[Epoch {epoch}] Test Accuracy: {test_acc:.4f}")
        wandb.log({"acc_test": test_acc, "epoch": epoch})
        
        torch.cuda.empty_cache()
        
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(args.savedir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save(m.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch+1}: {checkpoint_path}")

    savedir = os.path.join(args.savedir, str(f'experiment-{args.shadow_id}_{args.n_shadows}'))
    os.makedirs(savedir, exist_ok=True)
    np.save(savedir + "/keep.npy", keep_bool)
    torch.save(m.state_dict(), savedir + "/model.pt")

    
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

        
@torch.no_grad()
def get_acc(model, dl):
    model.eval()
    
    acc = []
    for x, y in dl:
        x, y = x.to(DEVICE), y.to(DEVICE)
        acc.append(torch.argmax(model(x), dim=1) == y)
    acc = torch.cat(acc)
    acc = torch.sum(acc) / len(acc)

    return acc.item()


if __name__ == "__main__":
    run()