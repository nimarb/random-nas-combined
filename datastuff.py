import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as dset
import numpy as np

from pathlib import Path


def get_distortion_tests(test_dir=None):
    if not test_dir:
        test_dir = '/ceph/blume/datasets/CIFAR10-C/test/'
        if not Path(test_dir).exists():
            test_dir = '/home/blume/datasets/CIFAR10-C/test/'
        if not Path(self.label_path).exists():
            self.label_path = '/home/suganuma/dataset/CIFAR10-C/test/'
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
        'zoom_bluy'
    ]
    return test_names


class NoisySet(Dataset):

    def __init__(self, test_path):
        self.test_path = test_path
        self.label_path = '/ceph/blume/datasets/CIFAR10-C/test/labels.npy'
        if not Path(self.label_path).exists():
            self.label_path = '/home/blume/datasets/CIFAR10-C/test/labels.npy'
        if not Path(self.label_path).exists():
            self.label_path = '/home/suganuma/dataset/CIFAR10-C/test/labels.npy'
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


def get_test_loader(test_path):
    dataloader = DataLoader(NoisySet(test_path), batch_size=128, shuffle=True,
                            # dataloader = DataLoader(dset.CIFAR10('./'), batch_size=128, shuffle=True,
                            num_workers=1, drop_last=True, pin_memory=True)
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
