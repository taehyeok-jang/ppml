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
from torch.utils.data import DataLoader, random_split, ConcatDataset
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

    From the paper, 
    
       data distribution: 
           D ~ _D (D ⊂ ~_D)
       
           challenger(victim model) is trained on D. 
      
       target point: 
           target point (x, y) ~_D -> IN (N/2) + OUT (N/2).
           that is, some target points ⊂ D, and others /⊂ D. 
       

    (1-0)
        load imagenet-1k 'train', and split by T1 + T2 + T3 (T1, T2, T3 must be always same indices across systems)
        load imagenet-1k 'val',   and split by S1 + S2      (S1, S2 must be always same indices) 
        (|T1|~=|S1|, |T2|~=|S2|) 


    (1-1) 
        split by,
        T1 -> T1 + T1' (T, T' can be different accross systems) 
        S1 -> S1 + S1' 

        T1'+S1' is used for the evaluation loop.


    (1-2) 
        to prepare datasets for training N shadow models, 

        create a partition for, 
            shadow 1: (T1+S1)_1
            shadow 2: (T1+S1)_2
            ... 
            shadow N: (T1+S1)_N

        the code is for a single shadow model. => save keep in file system.

    (1-3) 
        now we are ready to train. 

        train set: (T1+S1)_i
        eval set: (T1'+S1')


step 2: inference.py => 각 shadow models 별로 logits 출력하여 저장함.
step 3: score.py => 그냥 logit scaling

step 4: plot.py (eval) => membership inference attack to victim model. 

4-0. 
    load imagenet-1k 'train', and split by T1 + T2 + T3 (T1, T2, T3 must be always same indices across systems)
    load imagenet-1k 'val',   and split by S1 + S2      (S1, S2 must be always same indices) 

''' 

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
    test_ds = ConcatDataset([T_eval, S_eval])
    
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
        keep = keep.nonzero()[0] # (19910,) = (train samples * pkeep) 
    else:
        keep = np.random.choice(size, size=int(args.pkeep * size), replace=False)
        keep.sort()
    keep_bool = np.full((size), False)
    keep_bool[keep] = True

    train_ds = torch.utils.data.Subset(train_ds, keep)
    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4)
    test_dl = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=4)

    savedir = os.path.join(args.savedir, str(f'experiment-{args.shadow_id}_{args.n_shadows}'))
    os.makedirs(savedir, exist_ok=True)
    
    m = network(args.model, pretrained_=False)
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
            checkpoint_path = os.path.join(args.savedir, str(f'checkpoint-epoch_{epoch}.pt'))
            torch.save(m.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch+1}: {checkpoint_path}")

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
        return models.__dict__[arch](pretrained=pretrained_)
    elif arch in PYTORCH_IMAGE_MODELS:
        return timm.create_model(arch, pretrained=pretrained_)
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