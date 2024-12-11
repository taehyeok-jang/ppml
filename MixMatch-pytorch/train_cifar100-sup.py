import argparse
import os
import time
import datetime 
import random
from pathlib import Path

import numpy as np
import torch
import wandb
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset, random_split
import torch.utils.data as data # for code compatibility
from torchvision import models, transforms
from torchvision.datasets import CIFAR10, CIFAR100
import timm

import scipy.stats
from sklearn.metrics import auc, roc_curve

import pytorch_lightning as pl

# util
from tqdm import tqdm
from collections import Counter

# customized 
import models.arch as models

import dataset.cifar100 as dataset
# import dataset.cifar100_larger as dataset # for ViT model,

from utils import Logger, AverageMeter, accuracy, mkdir_p, savefig
from progress.bar import Bar as Bar
import utils as utils_
from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser(description='PyTorch MixMatch Training')
# Optimization options

parser.add_argument("--model", default="resnet18", type=str, metavar='N',
                    help='an adversary model that learns from labeled/unlabeled data')
parser.add_argument("--dataset", default="cifar100", type=str, metavar='N',
                    help='dataset')
parser.add_argument("--n-classes", default=100, type=int, metavar='N',
                    help='number of classes of dataset')


parser.add_argument('--lr', '--learning-rate', default=0.02, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')

# Checkpoints
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Miscs
parser.add_argument('--manualSeed', type=int, default=None, help='manual seed')
#Device options
parser.add_argument('--gpu', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
#Method options
parser.add_argument('--n-labeled', type=int, default=10000,
                        help='Number of labeled data')

parser.add_argument('--debug', default=False, type=bool)

# Searchable hyperparams 
parser.add_argument('--batch-size', default=64, type=int, metavar='N',
                    help='train batchsize')
# parser.add_argument('--train-iteration', type=int, default=1024,
#                         help='Number of iteration per epoch')

parser.add_argument('--alpha', default=0.75, type=float)    # default: 0.75
parser.add_argument('--lambda-u', default=20, type=float)   # default: 75

parser.add_argument('--T', default=0.5, type=float)
parser.add_argument('--ema-decay', default=0.999, type=float)   # default: 0.999

parser.add_argument('--out', default='',
                        help='Directory to output the result')

args = parser.parse_args()

args.n_unlabeled = 45000 - args.n_labeled
args.train_iteration = int(args.n_labeled / args.batch_size)                            # phase 1: fully-supervised learning
# args.train_iteration = int(max(args.n_labeled, args.n_unlabeled) / args.batch_size)   # phase 2: semi-supervised learning
##

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = 1583745484 # random.randint(1, 10000)

# Outputs
if args.out == '':
    args.out = 'experiments/{}_only/{}@{}_lr_{}_lu_{}'.format(
        args.model, 
        args.dataset, 
        args.n_labeled, 
        args.lr, 
        args.lambda_u
    )

state = {k: v for k, v in args._get_kwargs()}
best_acc = 0  # best test accuracy

def main():

    print(state)

    global best_acc

    if not os.path.isdir(args.out):
        mkdir_p(args.out)

    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)

    print(f'==> Preparing cifar100')
    transform_train = transforms.Compose([
        dataset.RandomPadandCrop(32),
        dataset.RandomFlip(),
        dataset.ToTensor(),
    ])

    transform_val = transforms.Compose([
        dataset.ToTensor(),
    ])
    datadir = Path().home() / "dataset"

    batch_size=args.batch_size

    # measure time for data pre-processing
    start_time = time.time()

    train_labeled_set, train_unlabeled_set, val_set, test_set = dataset.get_cifar100(datadir, args.n_labeled, transform_train=transform_train, transform_val=transform_val)
    labeled_trainloader = data.DataLoader(train_labeled_set, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    unlabeled_trainloader = data.DataLoader(train_unlabeled_set, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    val_loader = data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

    TEST_SAMPLES_RATIO = 0.2

    print(f'use {TEST_SAMPLES_RATIO * len(test_set):.0f} of {len(test_set)} for faster test/eval')
    """
    test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    """
    indices = np.random.choice(len(test_set), int(0.2 * len(test_set)), replace=False)
    subset_test_set = Subset(test_set, indices)
    test_loader = data.DataLoader(subset_test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    print(train_labeled_set)
    print(train_labeled_set[0][0].shape)

    print('create model, and ema_model')
    model = create_model()
    ema_model = create_model(ema=True)

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    ema_optim = WeigthEMA(model, ema_model, alpha=args.ema_decay)

    train_criterion = CustomLoss()
    criterion = nn.CrossEntropyLoss()

    if not os.path.isdir(args.out):
        mkdir_p(args.out)

    title = 'noisy-cifar-10'

    logger = Logger(os.path.join(args.out, 'log.txt'), title=title)
    logger.set_names(['Train Loss', 'Train Loss X', 'Train Loss U',  'Valid Loss', 'Valid Acc.', 'Test Loss', 'Test Acc.'])
    writer = SummaryWriter(args.out)

    best_acc = 0  # best test accuracy
    step = 0
    test_accs = []

    for epoch in range(args.epochs):

        train_loss, train_loss_x, train_loss_u = train(labeled_trainloader, unlabeled_trainloader, model, optim, ema_optim, train_criterion, epoch, use_cuda)
        """
        _, train_acc = validate(labeled_trainloader, ema_model, criterion, epoch, use_cuda, mode='Train Stats')
        val_loss, val_acc = validate(val_loader, ema_model, criterion, epoch, use_cuda, mode='Valid Stats')
        test_loss, test_acc = validate(test_loader, ema_model, criterion, epoch, use_cuda, mode='Test Stats ')
        """
        _, train_acc = 0.0, 0.0
        val_loss, val_acc = validate(val_loader, ema_model, criterion, epoch, use_cuda, mode='Valid Stats')
        test_loss, test_acc = validate(test_loader, model, criterion, epoch, use_cuda, mode='Test Stats ')

        sched.step()

        _test_acc = get_acc(model, test_loader)
        """
        print(f"[Epoch {epoch}] (naive) Test Accuracy: {_test_acc:.4f}")
        print(f"[Epoch {epoch}] Train Accuracy: {train_acc:.4f}")
        print(f"[Epoch {epoch}] Validation Accuracy: {val_acc:.4f}")
        print(f"[Epoch {epoch}] Test Accuracy: {test_acc:.4f}")
        """
        print(f"[Epoch {epoch}] (naive) Test Accuracy: {_test_acc:.4f}")
        print(f"[Epoch {epoch}] Validation Accuracy: {val_acc:.4f}")
        print(f"[Epoch {epoch}] Test Accuracy: {test_acc:.4f}")
        
        
        step = args.train_iteration * (epoch + 1)

        writer.add_scalar('losses/train_loss', train_loss, (epoch + 1))
        writer.add_scalar('losses/valid_loss', val_loss, (epoch + 1))
        writer.add_scalar('losses/test_loss', test_loss, (epoch + 1))

        writer.add_scalar('accuracy/train_acc', train_acc, (epoch + 1))
        writer.add_scalar('accuracy/val_acc', val_acc, (epoch + 1))
        writer.add_scalar('accuracy/test_acc', test_acc, (epoch + 1))

        logger.append([train_loss, train_loss_x, train_loss_u, val_loss, val_acc, test_loss, test_acc])

        best_acc = max(val_acc, best_acc)
        test_accs.append(test_acc)

    logger.close()
    writer.close()

    print('Best acc:')
    print(best_acc)

    print('Mean acc:')
    print(np.mean(test_accs[-20:]))


def create_model(ema=False):
    model = models.network(args.model, pretrained=True, n_classes=args.n_classes)
    model = model.cuda()

    if ema:
        for param in model.parameters():
            param.detach_()

    return model

def linear_rampup(current, rampup_length=args.epochs):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)
    
class WeigthEMA(object):
    def __init__(self, model, ema_model, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = 0.02 * args.lr # for lr=0.02, wd = 0.0004

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            if ema_param.dtype==torch.float32:
                ema_param.mul_(self.alpha)
                ema_param.add_(param * one_minus_alpha)
                # customized weight decay (TODO: should carefully set up)
                # 
                # lr=0.02 -> (0.9996)^5928 (1 epoch): 0.09332.
                # lr=0.002 -> (0.99996)^5928 (1 epoch): 0.78889
                # param.mul_(1 - self.wd) # x 0.9996 (for lr=0.02), x0.99996 (for lr=0

class CustomLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch):
        """
        probs_u = torch.softmax(outputs_u, dim=1)
        """
        
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        
        # CHECK
        """
        Lu = torch.mean((probs_u - targets_u)**2)
        w = _lambda_u * linear_rampup(epoch) # _lambda_u: 75 (default)
        """ 
        Lu = 0
        w = 0

        return Lx, Lu, 0
    
@torch.no_grad()
def get_acc(model, dl):
    acc = []
    for x, y in dl:
        if use_cuda:
            x, y = x.cuda(), y.cuda(non_blocking=True)
        acc.append(torch.argmax(model(x), dim=1) == y)
    acc = torch.cat(acc)
    acc = torch.sum(acc) / len(acc)

    return acc.item()

def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets

def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]

def train(labeled_trainloader, unlabeled_trainloader, model, optimizer, ema_optimizer, criterion, epoch, use_cuda):
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()
    ws = AverageMeter()
    end = time.time()
    
    bar = Bar('Training', max=args.train_iteration)
    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)
    
    model.train()

    with tqdm(range(args.train_iteration), desc="Training Progress", unit="batch") as progress_bar:
        for batch_idx in progress_bar:
            try:
                inputs_x, targets_x = next(labeled_train_iter)
            except:
                labeled_train_iter = iter(labeled_trainloader)
                inputs_x, targets_x = next(labeled_train_iter)

            """
            try:
                (inputs_u, inputs_u2), _ = next(unlabeled_train_iter) # two different augment(x)s;
            except:
                unlabeled_train_iter = iter(unlabeled_trainloader)
                (inputs_u, inputs_u2), _ = next(unlabeled_train_iter)
                """
    
            # measure data loading time
            data_time.update(time.time() - end)
            
            batch_size = inputs_x.size(0)
    
            # convert label to one-hot
            targets_x = torch.zeros(batch_size, args.n_classes).scatter_(1, targets_x.view(-1,1).long(), 1)
    
            if use_cuda:
                inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
                """
                inputs_u = inputs_u.cuda()
                inputs_u2 = inputs_u2.cuda()
                """

            """
            with torch.no_grad():
                # compute guessed labels of unlabel samples
                outputs_u = model(inputs_u)
                outputs_u2 = model(inputs_u2)
                p = (torch.softmax(outputs_u, dim=1) + torch.softmax(outputs_u2, dim=1)) / 2
                pt = p**(1/_T)
                targets_u = pt / pt.sum(dim=1, keepdim=True)
                targets_u = targets_u.detach()
            """
            
            # mixup 
            """ 
            all_inputs = torch.cat([inputs_x, inputs_u, inputs_u2], dim=0)
            all_targets = torch.cat([targets_x, targets_u, targets_u], dim=0)
            """
            all_inputs = torch.cat([inputs_x], dim=0)
            all_targets = torch.cat([targets_x], dim=0)

            # CHECK: 
            """
            l = np.random.beta(_alpha, _alpha)
            """
            l = 1
            l = max(l, 1-l)
            
            idx = torch.randperm(all_inputs.size(0))
            
            input_a, input_b = all_inputs, all_inputs[idx]
            target_a, target_b = all_targets, all_targets[idx]
            
            # input_a > input_b is guaranteed.
            mixed_input = l * input_a + (1 - l) * input_b
            mixed_target = l * target_a + (1 - l) * target_b

            """
            # interleave labeled and unlabed samples between batches to get correct batchnorm calculation 
            mixed_input = list(torch.split(mixed_input, batch_size))
            mixed_input = interleave(mixed_input, batch_size)
            """

            """
            logits = [model(mixed_input[0])]
            for input in mixed_input[1:]:
                logits.append(model(input))
            """
            logits = [model(mixed_input)]
    
            # put interleaved samples back
            """
            logits = interleave(logits, batch_size)
            logits_x = logits[0]
            logits_u = torch.cat(logits[1:], dim=0)
            """
            logits_x = logits[0]

            """
            Lx, Lu, w = criterion(logits_x, mixed_target[:batch_size], logits_u, mixed_target[batch_size:], epoch+batch_idx/_train_iteration)
            """
            Lx, Lu, w = criterion(logits_x, mixed_target[:batch_size], None, None, epoch+batch_idx/args.train_iteration)
    
            loss = Lx + w * Lu
    
            losses.update(loss, inputs_x.size(0))
            losses_x.update(Lx, inputs_x.size(0))
            losses_u.update(Lu, inputs_x.size(0))
            ws.update(w, inputs_x.size(0))
    
            optimizer.zero_grad()
            loss.backward()
            
            optimizer.step()
            # CHECK 
            # ema_optimizer.step()
    
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
        
            # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Loss_x: {loss_x:.4f} | Loss_u: {loss_u:.4f} | W: {w:.4f}'.format(
                        batch=batch_idx + 1,
                        size=args.train_iteration,
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg,
                        loss_x=losses_x.avg,
                        loss_u=losses_u.avg,
                        w=ws.avg,
                        )
            bar.next()

            progress_bar.set_postfix({
                "Data": f"{data_time.avg:.3f}s",
                "Batch": f"{batch_time.avg:.3f}s",
                "Total": bar.elapsed_td,
                "ETA": bar.eta_td,
                "Loss": f"{losses.avg:.4f}",
                "Loss_x": f"{losses_x.avg:.4f}",
                "Loss_u": f"{losses_u.avg:.4f}",
                "W": f"{ws.avg:.4f}",
            })
            
        bar.finish()
    
    return (losses.avg, losses_x.avg, losses_u.avg,)

def validate(valloader, model, criterion, epoch, use_cuda, mode):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar(f'{mode}', max=len(valloader))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            # measure data loading time
            data_time.update(time.time() - end)

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(valloader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                        )
            bar.next()
        bar.finish()
        
    return (losses.avg, top1.avg)


if __name__ == '__main__':
    main()
