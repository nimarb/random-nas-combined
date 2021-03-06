import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as dset
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import platform

from pathlib import Path


def get_distortion_tests(test_dir=None):
    if not test_dir:
        test_dir = '/home/blume/datasets/CIFAR10-C/test/'
        if 'nbpc' in platform.node():
            test_dir = '/home/nimar/progs/random-nas-combined/test-distortions/'
        elif 'yagi22' in platform.node() or 'yagi21' in platform.node():
            test_dir = '/home/suganuma/dataset/CIFAR10-C/test/'
        elif 'archtp480s' in platform.node():
            test_dir = '/home/nb/progs/random-nas-combined/test-distortions/'
    test_dists = [
        'brightness.npy',
        'contrast.npy',
        'defocus_blur.npy',
        'elastic_transform.npy',
        'fog.npy',
        'frost.npy',
        'gaussian_blur.npy',
        'gaussian_noise.npy',
        'glass_blur.npy',
        'impulse_noise.npy',
        'jpeg_compression.npy',
        # 'labels.npy',
        'motion_blur.npy',
        'pixelate.npy',
        'saturate.npy',
        'shot_noise.npy',
        'snow.npy',
        'spatter.npy',
        'speckle_noise.npy',
        'zoom_blur.npy'
    ]
    test_paths = []
    for test in test_dists:
        test_paths.append(test_dir + test)
    return test_paths


def get_distortion_tests_name():
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
    return test_names


class NoisySet(Dataset):

    def __init__(self, test_path):
        self.test_path = test_path
        self.label_path = '/home/blume/datasets/CIFAR10-C/test/labels.npy'
        if 'nbpc' in platform.node():
            self.label_path = '/home/nimar/progs/random-nas-combined/test-distortions/labels.npy'
        elif 'yagi22' in platform.node() or 'yagi21' in platform.node():
            self.label_path = '/home/suganuma/dataset/CIFAR10-C/test/labels.npy'
        elif 'archtp480s' in platform.node():
            self.label_path = '/home/nb/progs/random-nas-combined/test-distortions/labels.npy'
        # self.label_path = 'labels.npy'
        self.data = np.load(test_path)
        self.targets = np.load(self.label_path).tolist()
        self.app_transforms = transforms.Compose([
            # transforms.Scale(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768))
        ])

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img = self.data[idx]
        sample = self.app_transforms(img)
        return sample, self.targets[idx]


def get_test_loader(test_path, batch_size=128, num_test=50000):
    # To be able to test on the full CIFAR10 dataset
    if 50000 == num_test:
        balanced_valid_sampler = None
        shuffle = True
        balanced_valid_sampler = None
    else:
        print('./sample_indices/valid_sample_list_%d.npy' % (num_test))
        balanced_valid_indices = np.load(
            './sample_indices/valid_sample_list_%d.npy' % (num_test))
        balanced_valid_sampler = SubsetRandomSampler(
            balanced_valid_indices.tolist())
        shuffle = False
    dataloader = DataLoader(NoisySet(test_path), batch_size=batch_size,
                            shuffle=shuffle, num_workers=1, drop_last=True,
                            pin_memory=True, sampler=balanced_valid_sampler)
    return dataloader


def get_test_loader2():
    # dataloader = DataLoader(NoisySet(test_path), batch_size=128, shuffle=True,
    dataloader = DataLoader(dset.CIFAR10('./'), batch_size=128, shuffle=False,
                            num_workers=0, drop_last=True, pin_memory=True)
    return dataloader


if __name__ == "__main__":
    ldr = get_test_loader('brightness.npy')
    # ldr2 = get_test_loader2('brightness.npy')
    for _, (data, target) in enumerate(ldr):
        print('hi')
