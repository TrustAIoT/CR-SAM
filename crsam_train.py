import os
import time
import logging
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision.models import resnet18 as imagenet_resnet18
from torchvision.models import resnet50 as imagenet_resnet50
from torchvision.models import resnet101 as imagenet_resnet101
from models import cifar_resnet50, cifar_resnet18, cifar_resnet101, cifar_wrn28_10

from utils.crsam import CRSAM, enable_running_stats, disable_running_stats
from utils.dataset import CIFAR
from utils.metrics import accuracy
from utils.logger import CSVLogger, AverageMeter

from metrics.hessian import Hessian
from metrics.metrics import grad_norm, eigen_spec

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="cifar100", help='Dataset')
parser.add_argument('--model', default='resnet18')
parser.add_argument("--aug", default='basic', type=str, choices=['basic', 'cutout', 'autoaugment'], help='Data augmentation')
parser.add_argument('--epochs', type=int, default=200, help='Epochs')
parser.add_argument('--rho', type=float, default=0.1, help='rho parameter for SAM')
parser.add_argument('--gamma', type=float, default=0.1, help='gamma parameter for curvature regularizer')
parser.add_argument('--beta', type=float, default=0.01, help='beta parameter for curvature regularizer')
parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
parser.add_argument('--bs', type=int, default=128, help='batch size')
parser.add_argument('--mo', type=float, default=0.9, help='momentum')
parser.add_argument('--wd', type=float, default=1e-3, help='weight decay')
parser.add_argument('--print_freq', type=int, default=100)
parser.add_argument('--seed', type=int, default=0, help='seed')
parser.add_argument('--loadckpt', default=False, action='store_true')
args = parser.parse_args()

if args.dataset == 'cifar10':
    args.num_classes = 10
    args.milestones = [100, 120]
    args.data_dir = f"./data/{args.dataset}"
elif args.dataset == 'cifar100':
    args.num_classes = 100
    args.milestones = [100, 150]
    args.data_dir = f"./data/{args.dataset}"
elif args.dataset == 'imagenet':
    args.num_classes = 1000
    args.milestones = [30, 60, 90]
    args.data_dir = f"./data/{args.dataset}"
elif args.dataset == 'tinyimagenet':
    args.num_classes = 200
    args.milestones = [30, 60, 90]
    args.data_dir = f"./data/{args.dataset}"
else:
    print(f"BAD COMMAND dtype: {args.dataset}")

#random seed
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# Intialize directory and create path
args.ckpt_dir = "./results"
os.makedirs(args.ckpt_dir, exist_ok=True)
logger_name = os.path.join(args.ckpt_dir, f"crsam_{args.model}_{args.dataset}_{args.aug}_run{args.seed}")

# Logging tools
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler(logger_name + ".log"),
        logging.StreamHandler(),
    ],
)
logging.info(args)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#step = 1
def run_one_epoch(phase, loader, model, criterion, optimizer, args):
    #global step
    loss, acc = AverageMeter(), AverageMeter()
    t = time.time()

    for batch_idx, inp_data in enumerate(loader, 1):
        inputs, targets = inp_data
        inputs, targets = inputs.to(device), targets.to(device)

        if phase == 'train':
            model.train()
            with torch.set_grad_enabled(True):
                # compute output
                enable_running_stats(model)
                outputs = model(inputs)
                batch_loss = criterion(outputs, targets)
                optimizer.o_l = batch_loss

                # compute gradient 
                optimizer.zero_grad()
                batch_loss.backward(retain_graph=True)
                optimizer.first_step(zero_grad=True)

                # second forward-backward pass
                disable_running_stats(model)
                outputs = model(inputs)
                batch_loss = criterion(outputs, targets) 
                
                optimizer.w_l = batch_loss
                
                batch_loss.backward(retain_graph=True)
                optimizer.second_step(zero_grad=True)

                # second forward-backward pass
                enable_running_stats(model)
                outputs = model(inputs)
                batch_loss = criterion(outputs, targets) 
                
                optimizer.b_l = batch_loss
                
                batch_loss.backward()
                optimizer.third_step(zero_grad=True)

            #if step % 100 == 0:
            #    model.eval()
            #    hessian_comp = Hessian(model, criterion, data=(inputs, targets))
            #    top_eigenvalues, _ = hessian_comp.eigenvalues()
            #    trace = hessian_comp.trace()

            #    eigen_logger.save_values(step, top_eigenvalues, np.mean(trace), grad_norm_l2, optimizer.w_l.item(), optimizer.o_l.item(), optimizer.b_l.item())

            #step += 1
                
        elif phase == 'val':
            model.eval()
            with torch.no_grad():
                outputs = model(inputs)
                batch_loss = criterion(outputs, targets)
        else:
            logging.info('Define correct phase')
            quit()

        loss.update(batch_loss.item(), inputs.size(0))

        batch_acc = accuracy(outputs, targets, topk=(1,))[0]
        acc.update(float(batch_acc), inputs.size(0))

        if batch_idx % args.print_freq == 0:
            info = f"Phase:{phase} -- Batch_idx:{batch_idx}/{len(loader)}" \
                   f"-- {acc.count / (time.time() - t):.2f} samples/sec" \
                   f"-- Loss:{loss.avg:.2f} -- Acc:{acc.avg:.2f}"
            logging.info(info)

    return loss.avg, acc.avg


def main(args):
    dataset = CIFAR(args)
    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        if args.model == 'resnet50':
            model = cifar_resnet50(num_classes=args.num_classes)
        elif args.model == 'resnet18':
            model = cifar_resnet18(num_classes=args.num_classes)
        elif args.model == 'resnet101':
            model = cifar_resnet101(num_classes=args.num_classes)
        elif args.model == 'wrn':
            model = cifar_wrn28_10(num_classes=args.num_classes)
        else:
            print("define model")
            quit()
    elif 'imagenet' in args.dataset:
        if args.model == 'resnet50':
            model = imagenet_resnet50(num_classes=args.num_classes)
        elif args.model == 'resnet18':
            model = imagenet_resnet18(num_classes=args.num_classes)
        elif args.model == 'resnet101':
            model = imagenet_resnet101(num_classes=args.num_classes)
        else:
            print("define model")
            quit()
    else:
        print("define dataset type")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = CRSAM(model.parameters(), optim.SGD, rho=args.rho, gamma=args.gamma, beta=args.beta, lr=args.lr, momentum=args.mo, weight_decay=args.wd)

    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)
    base_optimizer = optimizer.base_optimizer
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(base_optimizer, T_max=args.epochs)
    
    csv_logger = CSVLogger(args, ['Epoch', 'Lr', 'Train Loss', 'Train Accuracy', 'Test Loss', 'Test Accuracy'], logger_name + '.csv')
    #eigen_logger = CSVLogger(args, ['Epoch', 'train_top_eigen', 'train_trace', 'train_grad_norm', 'test_top_eigen', 'test_trace', 'test_grad_norm'], logger_name + '_eigen.csv')

    if args.loadckpt:
        state = torch.load(f"{logger_name}_best.pth.tar")
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        best_acc = state['best_acc']
        start_epoch = state['epoch'] + 1
    else:
        start_epoch = 0
        best_acc = -float('inf')

    for epoch in range(start_epoch, args.epochs):
        logging.info('Epoch: [%d | %d]' % (epoch, args.epochs))

        trainloss, trainacc = run_one_epoch('train', dataset.train, model, criterion, optimizer, args)
        logging.info('Train_Loss = {0}, Train_acc = {1}'.format(trainloss, trainacc))
        
        valloss, valacc = run_one_epoch('val', dataset.test, model, criterion, optimizer, args)
        logging.info('Val_Loss = {0}, Val_acc = {1}'.format(valloss, valacc))

        lr = scheduler.optimizer.param_groups[0]['lr']
        
        csv_logger.save_values(epoch, lr, trainloss, trainacc, valloss, valacc)
        
        #if epoch % 10 == 0:
        #    train_grad_norm = grad_norm(model, criterion, optimizer, dataloader=dataset.train, lp=2)
        #    train_top_eigen, train_trace = eigen_spec(model, criterion, dataloader=dataset.train)
        #    test_grad_norm = grad_norm(model, criterion, optimizer, dataloader=dataset.test, lp=2)
        #    test_top_eigen, test_trace = eigen_spec(model, criterion, dataloader=dataset.test)
        #    eigen_logger.save_values(epoch, train_top_eigen, train_trace, train_grad_norm, test_top_eigen, test_trace, test_grad_norm)

        scheduler.step()

        if valacc > best_acc:
            state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'best_acc': best_acc
            }
            torch.save(state, f"{logger_name}_best.pth.tar")
            best_acc = valacc
        logging.info(f'best acc:{best_acc}')

        #if epoch % 100 == 0:
        #    state = {
        #        'model': model.state_dict(),
        #        'optimizer': optimizer.state_dict(),
        #        'scheduler': scheduler.state_dict(),
        #        'epoch': epoch,
        #        'best_acc': best_acc
        #    }
        #    torch.save(state, f"{logger_name}_epoch_{epoch}.pth.tar")

if __name__ == '__main__':
    main(args)
