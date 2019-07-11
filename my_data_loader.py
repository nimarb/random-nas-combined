"""
[1]: https://discuss.pytorch.org/t/feedback-on-pytorch-for-kaggle-competitions/2252/4
[2]: https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb
"""

import torch
import numpy as np

from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
import utils


class FromIndexSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, indices, max_sample_num=None):
        self.indices = indices
        self.max_sample_num = int(max_sample_num)

    def __iter__(self):
        if not self.max_sample_num:
            return iter(self.indices)
        else:
            return iter(self.indices[:self.max_sample_num])

    def __len__(self):
        if not self.max_sample_num:
            return len(self.indices)
        else:
            return self.max_sample_num


def get_train_valid_loader(data_dir, batch_size, augment, random_seed, valid_size=0.1, shuffle=True, show_sample=False, num_workers=4, pin_memory=False,
                           data_num=500):

    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    train_transform, valid_transform = utils._data_transforms_cifar10(
        False, 16)

    # load the dataset
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform)
    valid_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=valid_transform)

    print('./sample_indices/training_sample_list_%d.npy' % (data_num))
    print('./sample_indices/valid_sample_list_%d.npy' % (data_num))
    balanced_train_indices = np.load(
        './sample_indices/training_sample_list_%d.npy' % (data_num))
    balanced_train_sampler = SubsetRandomSampler(
        balanced_train_indices.tolist())
    balanced_valid_indices = np.load(
        './sample_indices/valid_sample_list_%d.npy' % (data_num))
    balanced_valid_sampler = SubsetRandomSampler(
        balanced_valid_indices.tolist())

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        sampler=balanced_train_sampler)

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        sampler=balanced_valid_sampler)

    # # visualize some images
    # data_iter = iter(train_loader)
    # images, labels = data_iter.next()
    # X = images.numpy().transpose([0, 2, 3, 1])
    # plot_images(X, labels)

    # data_iter = iter(valid_loader)
    # images, labels = data_iter.next()
    # X = images.numpy().transpose([0, 2, 3, 1])
    # plot_images(X, labels)

    return (train_loader, valid_loader)


def get_train_valid_loader_tinyimagenet(data_dir, batch_size, augment, random_seed, valid_size=0.1, shuffle=True, show_sample=False, num_workers=4, pin_memory=False,
                                        data_num=20000):

    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    transform_train = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    # load the dataset
    train_dataset = torchvision.datasets.ImageFolder(
        root='/home/suganuma/dataset/tiny-imagenet-200/train', transform=transform_train)
    testset = torchvision.datasets.ImageFolder(
        root='/home/suganuma/dataset/tiny-imagenet-200/train', transform=transform_test)

    print('./sample_indices/training_sample_list_%d.npy' % (data_num))
    print('./sample_indices/valid_sample_list_%d.npy' % (data_num))
    balanced_train_indices = np.load(
        './sample_indices/training_sample_list_%d.npy' % (data_num))
    balanced_train_sampler = SubsetRandomSampler(
        balanced_train_indices.tolist())
    balanced_valid_indices = np.load(
        './sample_indices/valid_sample_list_%d.npy' % (data_num))
    balanced_valid_sampler = SubsetRandomSampler(
        balanced_valid_indices.tolist())

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        sampler=balanced_train_sampler)

    valid_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        sampler=balanced_valid_sampler)

    # # visualize some images
    # data_iter = iter(train_loader)
    # images, labels = data_iter.next()
    # X = images.numpy().transpose([0, 2, 3, 1])
    # plot_images(X, labels)

    # data_iter = iter(valid_loader)
    # images, labels = data_iter.next()
    # X = images.numpy().transpose([0, 2, 3, 1])
    # plot_images(X, labels)

    return (train_loader, valid_loader)


def get_test_loader(data_dir,
                    batch_size,
                    shuffle=True,
                    num_workers=0,
                    pin_memory=True):
    """
    Utility function for loading and returning a multi-process
    test iterator over the CIFAR-10 dataset.

    If using CUDA, num_workers should be set to 1 and pin_memory to True.

    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - shuffle: whether to shuffle the dataset after every epoch.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.

    Returns
    -------
    - data_loader: test set iterator.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    # define transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    dataset = datasets.CIFAR10(
        root=data_dir, train=False,
        download=True, transform=transform,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader
