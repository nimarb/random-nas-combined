#!/usr/bin/env python3

import time
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable

from cnn_model import CGP2CNN
from my_data_loader import get_train_valid_loader, get_train_valid_loader_tinyimagenet
from utils import Cutout
import torchvision

from datastuff import get_test_loader, get_distortion_tests


# __init__: load dataset
# __call__: training the CNN defined by CGP list
class CNN_train():
    def __init__(self, dataset_name, validation=True, verbose=True,
                 img_size=32, batchsize=128, data_num=500, mode="full",
                 config=None):
        self.verbose = verbose
        self.img_size = img_size
        self.validation = validation
        self.batchsize = batchsize
        self.dataset_name = dataset_name
        self.data_num = data_num
        self.mode = mode
        self.config = config

        # load dataset
        if dataset_name == 'cifar10' or dataset_name == 'tinyimagenet':
            if dataset_name == 'cifar10':
                self.n_class = 10
                self.channel = 3
                if self.validation:
                    self.dataloader, self.test_dataloader = get_train_valid_loader(
                        data_dir='./', batch_size=self.batchsize, augment=True, random_seed=2018, num_workers=1, pin_memory=True, data_num=self.data_num)
                else:
                    train_dataset = dset.CIFAR10(root='./', train=True, download=True,
                                                 transform=transforms.Compose([
                                                     transforms.RandomCrop(
                                                         32, padding=4),
                                                     transforms.RandomHorizontalFlip(),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize(
                                                         (0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768)),
                                                     Cutout(16),
                                                 ]))
                    test_dataset = dset.CIFAR10(root='./', train=False, download=True,
                                                transform=transforms.Compose([
                                                    # transforms.Scale(self.img_size),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(
                                                        (0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768)),
                                                ]))
                    self.dataloader = torch.utils.data.DataLoader(
                        train_dataset, batch_size=self.batchsize, shuffle=True, num_workers=int(4), drop_last=True)
                    self.test_dataloader = torch.utils.data.DataLoader(
                        test_dataset, batch_size=self.batchsize, shuffle=True, num_workers=int(4), drop_last=True)
            elif dataset_name == 'tinyimagenet':
                self.n_class = 200
                self.channel = 3
                if self.validation:
                    self.dataloader, self.test_dataloader = get_train_valid_loader_tinyimagenet(
                        data_dir='/home/suganuma/dataset/tiny-imagenet-200/train', batch_size=self.batchsize, augment=True, random_seed=2018, num_workers=4, pin_memory=False, data_num=self.data_num)
                else:
                    if self.mode == "full":
                        transform_train = transforms.Compose([
                            transforms.RandomCrop(64, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize(
                                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                            Cutout(16), ])

                        trainset = torchvision.datasets.ImageFolder(
                            root='/home/suganuma/dataset/tiny-imagenet-200/train', transform=transform_train)
                        self.dataloader = torch.utils.data.DataLoader(
                            trainset, batch_size=self.batchsize, shuffle=True, num_workers=8, drop_last=True)
                    else:
                        self.dataloader, _ = get_train_valid_loader_tinyimagenet(
                            data_dir='/home/suganuma/dataset/tiny-imagenet-200/train', batch_size=self.batchsize, augment=True, random_seed=2018, num_workers=4, pin_memory=False, data_num=self.data_num)
                        print("train  num", self.data_num)

                    transform_test = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
                    testset = torchvision.datasets.ImageFolder(
                        root='/home/suganuma/dataset/tiny-imagenet-200/val', transform=transform_test)
                    self.test_dataloader = torch.utils.data.DataLoader(
                        testset, batch_size=self.batchsize, shuffle=False, num_workers=4, drop_last=True)
        else:
            print('\tInvalid input dataset name at CNN_train()')
            exit(1)

    def __call__(self, cgp, gpuID, num_epoch=30, out_model='mymodel.model'):
        if self.verbose:
            print('GPUID     :', gpuID)
            print('num_epoch :', num_epoch)
            print('batch_size:', self.batchsize)
            print('data_num:', self.data_num)

        # model
        torch.backends.cudnn.benchmark = True
        model = CGP2CNN(cgp, self.channel, self.n_class, self.img_size,
                        arch_type=self.config['arch_type'])
        # model = nn.DataParallel(model)
        model = model.cuda(gpuID)
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.cuda(gpuID)
        optimizer = optim.Adam(
            model.parameters(), lr=0.001, betas=(0.5, 0.999))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, float(num_epoch))

        # Train loop
        for epoch in range(1, num_epoch+1):
            print('epoch', epoch)
            scheduler.step()
            start_time = time.time()
            train_loss = 0
            total = 0
            correct = 0
            model.train()
            for _, (data, target) in enumerate(self.dataloader):
                data = Variable(data, requires_grad=False).cuda(gpuID)
                target = Variable(target, requires_grad=False).cuda(gpuID)
                optimizer.zero_grad()
                try:
                    logits = model(data)
                except:
                    import traceback
                    traceback.print_exc()
                    return 0.
                loss = criterion(logits, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                _, predicted = logits.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
            print('Train set : Average loss: {:.4f}'.format(train_loss))
            print('Train set : Average Acc : {:.4f}'.format(correct/total))
            print('time ', time.time()-start_time)
            if self.validation:
                if epoch == num_epoch:
                    t_acc = self.test(model, criterion, gpuID)
            else:
                if epoch % 10 == 0:
                    t_acc = self.test(model, criterion, gpuID)
        # save the model
        torch.save(model.state_dict(), f"{self.config['save_dir']}model_0.pth")
        t_acc, accs = self.test_all(model, criterion, gpuID)
        return t_acc, accs

    def test_all(self, model, criterion, gpuID):
        accs = {}
        test_names = [
            'brightness',
            'contrast',
            'defocus_blur',
            'elastic_transform',
            'fog',
            'frost',
            'gaussian_blur',
            'gaussian_noise',
            'glass_blur',
            'impulse_noise',
            'jpeg_compression',
            # 'labels',
            'motion_blur',
            'pixelate',
            'saturate',
            'shot_noise',
            'snow',
            'spatter',
            'speckle_noise',
            'zoom_blur'
        ]
        test_dists = get_distortion_tests()
        dloader = self.test_dataloader
        acc = self.test(model, criterion, gpuID)
        accs['normal'] = acc
        for (idx, test_file) in enumerate(test_dists):
            self.test_dataloader = get_test_loader(test_file)
            acc = self.test(model, criterion, gpuID)
            accs[test_names[idx]] = acc

        self.test_dataloader = dloader
        return accs['normal'], accs

    # For validation/test
    def test(self, model, criterion, gpuID):
        total = 0
        correct = 0
        class_correct = list(0. for i in range(self.n_class))
        class_total = list(0. for i in range(self.n_class))
        acc_list = [0] * (self.n_class+1)
        model.eval()
        with torch.no_grad():
            for _, (data, target) in enumerate(self.test_dataloader):
                data = Variable(data, requires_grad=False).cuda(gpuID)
                target = Variable(target, requires_grad=False).cuda(gpuID)
                try:
                    logits = model(data)
                except:
                    import traceback
                    traceback.print_exc()
                    return 0.
                _, predicted = logits.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                c = (predicted == target).squeeze()
                for i in range(self.batchsize):
                    label = target[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
        print('Accuracy of the network on the test images:     %d %% (%d / %d)' %
              (100 * correct / total, correct, total))
        # for i in range(self.n_class):
        # acc_list[i] = 100 * class_correct[i] / class_total[i]
        # print('Accuracy of %d: (%d/%d)' % (i, class_correct[i], class_total[i]))
        # print('Accuracy of %d: %2d %% (%d/%d)' % (i, 100 * class_correct[i] / class_total[i], class_correct[i], class_total[i]))
        # print('Test set : (%d/%d)' % (correct, total))
        # print('Test set : Average Acc : {:.4f}'.format(correct/total))

        return (correct/total)
