import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from utils.cutout import Cutout
from utils.autoaugment import AutoAugment


class CIFAR:
    def __init__(self, args):
        mean, std = np.array([125.3, 123.0, 113.9]) / 255.0, np.array([63.0, 62.1, 66.7]) / 255.0

        train_transform = [
            torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
            torchvision.transforms.RandomHorizontalFlip()
        ]

        if args.aug == 'autoaugment':
            train_transform.append(AutoAugment())

        train_transform.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            ])
        
        if args.aug == 'cutout':
            train_transform.append(Cutout())
        
        train_transform = transforms.Compose(train_transform)

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        if args.dataset == 'cifar10':
            train_set = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True, download=True, transform=train_transform)
            test_set = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False, download=True, transform=test_transform)
        else:
            train_set = torchvision.datasets.CIFAR100(root='./data/cifar100', train=True, download=True, transform=train_transform)
            test_set = torchvision.datasets.CIFAR100(root='./data/cifar100', train=False, download=True, transform=test_transform)

        self.train = torch.utils.data.DataLoader(train_set, batch_size=args.bs, shuffle=True, pin_memory=True)
        self.test = torch.utils.data.DataLoader(test_set, batch_size=args.bs, shuffle=False, pin_memory=True)


