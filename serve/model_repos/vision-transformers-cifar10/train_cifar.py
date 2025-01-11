# -*- coding: utf-8 -*-
'''

Train CIFAR10 with PyTorch and Vision Transformers!
written by @kentaroy47, @arutema47

'''

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np

import torchvision
import torchvision.transforms as transforms

import timm

import sys 
import os
import argparse
import pandas as pd
import csv
import time

from models import *
from utils import progress_bar
from randomaug import RandAugment
from models.vit import ViT
from models.convmixer import ConvMixer

from tensorboardX import SummaryWriter

# parsers
parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--dataset', default="none", help="you must specify either cifar10 or cifar100")
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate') # resnets.. 1e-3, Vit..1e-4
parser.add_argument('--opt', default="adam")
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--noaug', action='store_true', help='disable use randomaug')
parser.add_argument('--noamp', action='store_true', help='disable mixed precision training. for older pytorch versions')
parser.add_argument('--nowandb', action='store_true', help='disable wandb')
parser.add_argument('--mixup', action='store_true', help='add mixup augumentations')
parser.add_argument('--net', default='vit_base_ptach16_384')
parser.add_argument('--dp', action='store_true', help='use data parallel')
parser.add_argument('--bs', default='128')
parser.add_argument('--size', default="32")
parser.add_argument('--n_epochs', type=int, default='200')
parser.add_argument('--patch', default='4', type=int, help="patch for ViT")
parser.add_argument('--dimhead', default="64", type=int)
parser.add_argument('--convkernel', default='8', type=int, help="parameter for convmixer")
parser.add_argument('--gpu', default=0, type=int, help="id(s) for CUDA_VISIBLE_DEVICES")


args = parser.parse_args()


if args.dataset == "none":
    print("Error: --dataset argument must be specified. Use --dataset <dataset_name>")
    sys.exit(1)


os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

# take in args
usewandb = ~args.nowandb
if usewandb:
    import wandb
    watermark = "{}_lr{}".format(args.net, args.lr)
    wandb.init(project=f"{args.dataset}-challange",
            name=watermark)
    wandb.config.update(args)

bs = int(args.bs)
imsize = int(args.size)

use_amp = not args.noamp
aug = args.noaug

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

'''
'''
print("==> Hyperparameters:")
print(f"Resume: {args.resume}")
print(f"Model: {args.net}")
print(f"Dataset: {args.dataset}")
print(f"GPU: {device + '_' + str(args.gpu)}")
print(f"Learning Rate: {args.lr}")
print(f"Optimizer: {args.opt}")
print(f"Batch Size: {args.bs}")
print(f"Number of Epochs: {args.n_epochs}")
print(f"Patch Size (ViT): {args.patch}")
print(f"Dimension Head: {args.dimhead}")
print(f"ConvKernel (ConvMixer): {args.convkernel}")
print(f"Data Augmentation Enabled: {aug}")
print(f"Mixed Precision Training (AMP): {use_amp}")
print(f"Device: {device}")
print(f"Network Architecture: {args.net}")
print(f"Data Parallel Enabled: {args.dp}")
print(f"Image Size: {imsize}")
print("="*50)

# Data
print('==> Preparing data..')

size = 384

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2023, 0.1994, 0.2010)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)

if args.dataset == "cifar10":
    _mean = cifar10_mean
    _std = cifar10_std
elif args.dataset == "cifar100":
    _mean = cifar100_mean
    _std = cifar100_std
else: 
    raise ValueError(f"Unsupported dataset: {args.dataset}")

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.Resize(size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(_mean, _std),
])

transform_test = transforms.Compose([
    transforms.Resize(size),
    transforms.ToTensor(),
    transforms.Normalize(_mean, _std),
])

# Add RandAugment with N, M(hyperparameter)
if aug:  
    N = 2; M = 14;
    transform_train.transforms.insert(0, RandAugment(N, M))

# Prepare dataset
if args.dataset == "cifar10":
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=8)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)
elif args.dataset == "cifar100":
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=8)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)
else: 
    raise ValueError(f"Unsupported dataset: {args.dataset}")


# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model factory..
print('==> Building model..')
if args.dataset == "cifar10":
    n_classes = 10 
elif args.dataset == "cifar100":
    n_classes = 100
else: 
    raise ValueError(f"Unsupported dataset: {args.dataset}")

# 
net = timm.create_model(args.net, pretrained=True)
if args.net.startswith("vit"):
    net.head = nn.Linear(net.head.in_features, n_classes)
elif args.net.startswith("convnext"):
    net.head.fc = nn.Linear(net.head.fc.in_features, n_classes)
else: 
    raise ValueError(f"Unsupported network: {args.net}")

# For Multi-GPU
if 'cuda' in device:
    print(device)
    if args.dp:
        print("using data parallel")
        net = torch.nn.DataParallel(net) # make parallel
        cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/{}/{}-4-ckpt.t7'.format(args.dataset, args.net))

    net.load_state_dict(checkpoint['model'])
    # best_acc = checkpoint['acc']
    # start_epoch = checkpoint['epoch']

# Loss is CE
criterion = nn.CrossEntropyLoss()

if args.opt == "adam":
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
elif args.opt == "sgd":
    optimizer = optim.SGD(net.parameters(), lr=args.lr)  
    
# use cosine scheduling
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs)

##### Training
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        # Train with amp
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = net(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | LR: %.6f'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total, optimizer.param_groups[0]['lr']))
    return train_loss/(batch_idx+1), 100.*correct/total

##### Validation
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            "model": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "acc": acc, 
            "epoch": epoch
            }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/{}/{}-4-ckpt.t7'.format(args.dataset, args.net))
        best_acc = acc
    
    os.makedirs("log", exist_ok=True)
    content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, val loss: {test_loss:.5f}, acc: {(acc):.5f}'
    print(content)
    with open(f'log/log_{args.net}_patch{args.patch}.txt', 'a') as appender:
        appender.write(content + "\n")
    return test_loss, acc


_out = 'tensorboard/{}/{}_lr_{}'.format(args.dataset, args.net, args.lr)
if not os.path.isdir(_out):
    os.makedirs(_out)

list_loss = []
list_acc = []

if usewandb:
    wandb.watch(net)

writer = SummaryWriter(_out)

net.cuda()
for epoch in range(start_epoch, args.n_epochs):
    start = time.time()
    train_loss, train_acc = train(epoch)
    val_loss, val_acc = test(epoch)
    
    scheduler.step(epoch-1) # step cosine scheduling
    
    list_loss.append(val_loss)
    list_acc.append(val_acc)

    
    writer.add_scalar('losses/train_loss', train_loss, epoch)
    writer.add_scalar('losses/val_loss', val_loss, epoch)

    writer.add_scalar('accuracy/train_acc', train_acc, epoch)
    writer.add_scalar('accuracy/val_acc', val_acc, epoch)
    
    # Log training..
    if usewandb:
        wandb.log({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss, "val_acc": val_acc, "lr": optimizer.param_groups[0]["lr"],
        "epoch_time": time.time()-start})

    # Write out csv..
    with open(f'log/log_{args.net}_patch{args.patch}.csv', 'w') as f:
        csv_writer = csv.writer(f, lineterminator='\n')
        csv_writer.writerow(list_loss) 
        csv_writer.writerow(list_acc) 
    print(list_loss)

# writeout wandb
if usewandb:
    wandb.save("wandb_{}.h5".format(args.net))

writer.close()
    
