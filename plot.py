import argparse
import matplotlib
import numpy as np
import seaborn as sns
import json
import csv

from matplotlib import pyplot as plt
from pathlib import Path
from datastuff import get_distortion_tests_name, get_distortion_tests, get_test_loader2
from create_consolidated_result_json import get_arch_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default='save_dir')
    parser.add_argument('--num_trains', type=int, nargs='+', default=[5000, 10000])
    parser.add_argument('--arch_type', type=str, default='densenet')
    args = parser.parse_args()
    return args


def get_num_train_acc_arch_dict_old(logdir='save_dir', net_type='resnet',
                                    num_train=1000,
                                    base_date='2019-06-11-03-37'):
    """ Looks like:
        arch_dict[configs]
        arch_dict['accs'] = []
        arch_dict['archs'] = []"""
    if isinstance(logdir, str):
        logdir = Path(logdir)

    base_str = f'{net_type}-{base_date}-*-{num_train}-*'
    acc_paths = logdir.glob(f'{base_str}/accuracy0.txt')
    config_paths = logdir.glob(f'{base_str}/config.json')
    arch_paths = logdir.glob(f'{base_str}/log.txt')

    arch_dict = {}
    accs = []
    archs = []

    for acc_p in acc_paths:
        with open(acc_p, 'r') as acc_f:
            lines = acc_f.readlines()
            for line in lines:
                accs.append(float(line))

    for arch_p in arch_paths:
        with open(arch_p, 'r') as arch_f:
            lines = arch_f.readlines()
            for line in lines:
                idx = line.index(',,')
                archs.append(line[idx+2:-1])

    for cfg in config_paths:
        with open(cfg, 'r') as c:
            cfgdata = json.load(c)
        break
    arch_dict = cfgdata
    arch_dict['accs'] = accs
    arch_dict['num_train'] = num_train
    arch_dict['avg_acc'] = sum(accs) / len(accs)
    arch_dict['max_acc'] = max(accs)
    arch_dict['archs'] = archs
    return arch_dict


def get_num_train_acc_arch_dict(logdir='save_dir', net_type='resnet',
                                num_train=1000, acc_type='normal',
                                base_date='2019-07-25'):
    """ Looks like:
        arch_dict[configs]
        arch_dict['accs'] = []
        arch_dict['archs'] = []"""
    if isinstance(logdir, str):
        logdir = Path(logdir)

    base_str = f'{net_type}-{base_date}-*-{num_train}-*'
    acc_paths = logdir.glob(f'{base_str}/accuracies.json')
    config_paths = logdir.glob(f'{base_str}/config.json')
    arch_paths = logdir.glob(f'{base_str}/log-active.txt')

    arch_dict = {}
    acc_dict = {}
    accs = []
    archs = []

    acc_ctr = 0
    for acc_p in acc_paths:
        with open(acc_p, 'r') as acc_f:
            acc = json.load(acc_f)
            for _, value in acc.items():
                accs.append(float(value[acc_type]))
                acc_dict[acc_ctr] = value

    for arch_p in arch_paths:
        with open(arch_p, 'r') as arch_f:
            lines = arch_f.readlines()
            for line in lines:
                idx = line.index(',,')
                archs.append(line[idx+2:-1])

    for cfg in config_paths:
        with open(cfg, 'r') as c:
            cfgdata = json.load(c)
        break
    arch_dict = cfgdata
    arch_dict['accs'] = accs
    arch_dict['num_train'] = num_train
    arch_dict['avg_acc'] = sum(accs) / len(accs)
    arch_dict['max_acc'] = max(accs)
    arch_dict['archs'] = archs
    return arch_dict


def get_imgs(nr=0):
    paths = get_distortion_tests('test-distortions/')
    # imgs = np.zeros(len(paths))
    imgs = []
    for idx, path in enumerate(paths):
        img = np.load(path)
        # imgs[idx] = img[nr]
        imgs.append(img[nr])
    return imgs


# def get_img_index(img):


def to_subplot(axes, picture, title):
    # axes.axis('off')
    axes.imshow(picture, cmap='gray', interpolation='nearest')
    axes.set_title(title, fontsize=30)
    axes.set_yticklabels([])
    axes.set_xticklabels([])
    #axes.xticks([], [])
    #axes.yticks([], [])


def plot_influence(arch_string, s_tests_id=0, title=None):
    """DOCS"""
    fig, axes = plt.subplots(nrows=4, ncols=5)

    # for id, ax in enumerate(axes[:, 0]):
    #     if id > 1:
    #         ax.set_ylabel('Harmful             ', rotation=0, size='large')
    #     else:
    #         ax.set_ylabel('Helpful             ', rotation=0, size='large')

    imgs = get_imgs()

    to_subplot(
        axes[0, 0],
        [[],[],[]],
        # get_test_loader2().dataset.data[int(s_tests_id)],
        # 'normal')
        '')
    # for i, ii, iii in [[0, 1, 0], [0, 2, 1], [0, 3, 2], [1, 0, 3], [1, 1, 4], [1, 2, 5], [1, 3, 6]]:
    for i, ii, iii in [[0, 1, 0], [0, 2, 1], [0, 3, 2], [0, 4, 7], [1, 0, 3], [1, 1, 4], [1, 2, 5], [1, 3, 6], [1, 4, 8]]:
        to_subplot(
            axes[i, ii],
            imgs[iii],
            f'{get_distortion_tests_name()[iii]}')
    for i, ii, iii in [[2, 0, 9], [2, 1, 10], [2, 2, 11], [2, 3, 12], [2, 4, 13], [3, 0, 14], [3, 1, 15], [3, 2, 16], [3, 3, 17], [3, 4, 18]]:
        to_subplot(
            axes[i, ii],
            imgs[iii],
            f'{get_distortion_tests_name()[iii]}')

    if not title:
        fig.suptitle(
            f"CIFAR10 test dataset postprocessed with various distortions",
            fontsize=45)
    else:
        # fig.suptitle(title, size='x-large')
        fig.suptitle(title, fontsize=45)
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None,
                        wspace=-0.2, hspace=0.4)
    fig.tight_layout()
    inf_fig_folder = "figures/influences/inf_"
    fig.set_figheight(13)
    fig.set_figwidth(13)
    # if recursion_depth and r_avg:
        # fig_fn = f'{inf_fig_folder}{arch_string}_rec-dep{recursion_depth}_r-avg{r_avg}_{get_cifar10_class(class_nr)}_{s_tests_id}.svg'
        # fig_fn = f'{inf_fig_folder}{arch_string}_rec-dep{recursion_depth}_r-avg{r_avg}_{get_cifar10_class(class_nr)}_{s_tests_id}.png'
    # else:
        # fig_fn = f'{inf_fig_folder}{arch_string}_{get_cifar10_class(class_nr)}_{s_tests_id}.svg'
        # fig_fn = f'{inf_fig_folder}{arch_string}_{get_cifar10_class(class_nr)}_{s_tests_id}.png'

    # print(f'Saved to: {fig_fn}')

    # fig.savefig(fig_fn, figsize=(7, 7), dpi=150)
    plt.show()


if __name__ == "__main__":
    args = parse_args()
    arch_dicts = []
    # args.num_trains = [500, 1000, 5000, 10000, 25000]
    accs = {}
    network_type = 'densenet'
    arch_d = get_arch_dict(net_type=network_type, base_date='2019')
    types = get_distortion_tests_name()
    for num_train in args.num_trains:
        for typ in types:
            d = get_num_train_acc_arch_dict(num_train=num_train, acc_type=typ)
            accs[typ] = d['avg_acc']
        arch_dicts.append(d)

        # with open(f'vgg-per_type.csv', 'a') as fout:
            # wrtr = csv.writer(fout)
            # wrtr = csv.DictWriter(fout, fieldnames=types)
            # wrtr.writeheader()
            # wrtr.writerow(accs)

            # wrtr.writerow([i for i in range(len(d['accs']))])
            # wrtr.writerow(lay_num)
    # plot_influence('hi')
    with open(f'{network_type}.json', 'w+') as fout:
        json.dump(arch_d, fout, indent=2)
