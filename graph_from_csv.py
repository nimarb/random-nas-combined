#%%

import numpy as np
import csv
import json
import argparse
import seaborn as sns
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt

sns.set(style='whitegrid')
sns.set_context("paper")

#%%
with open('resnet.json', 'r') as fin:
    resnet_d = json.load(fin)
with open('vgg.json', 'r') as fin:
    vgg_d = json.load(fin)
with open('densenet.json', 'r') as fin:
    densenet_d = json.load(fin)

test_dists = [
    'normal',
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

num_trains = [500, 1000, 5000, 10000, 25000]

def analyse_arch_dict(arch_dict, num_train=1000, distortion='normal',
                      plot_depths=False, show_graph=True, resnet_limit=False):
    depths = []
    normal_accs = []
    for _, value in arch_dict.items():
        if value['num_train'] == num_train:
            if resnet_limit:
                if value['num_conv_layers'] <= resnet_limit:
                    depths.append(value['num_conv_layers'])
                    normal_accs.append(value['accuracies'][distortion])
            else:
                depths.append(value['num_conv_layers'])
                normal_accs.append(value['accuracies'][distortion])

    if plot_depths:
        sns.lineplot(x=(range(len(depths))), y=depths)
        plt.show()
    print(f'max val: {max(depths)}')
    print(f'min val: {min(depths)}')
    print(f'avg val: {sum(depths)/len(depths)}')
    sorted_depth = sorted(depths)
    sns_plt = sns.lineplot(x=depths, y=normal_accs)
    title = f'ResNet Test Accuracy for No Distortions'
    sns_plt.legend(title='Num Train', labels=[str(i) for i in num_trains])
    sns_plt.set(xlabel='Num Conv Layers', ylabel='Test/Acc', title=title)
    save_graph(plt, title.replace(' ', '-').lower())
    if show_graph:
        plt.show()
    return plt, sns_plt


def compare_dists(arch_dict, main_distortion='normal',
                      plot_depths=False, show_graph=True, resnet_limit=False):
    
    # depths = {}  # contains one num_train entry for each trail
    # accs = {}  # contains one acc entry for each trail per dist
    accs2 = {}  # contains the avg acc for each unique num_train entry per dist
    for dist in test_dists:
        # accs[dist] = []
        accs2[dist] = []
        # depths[dist] = []
        for num_train in num_trains:
            acc_cnt = 0
            acc_total_current_num_train = 0
            for _, value in arch_dict.items():
                if value['num_train'] == num_train:
                    # depths[dist].append(value['num_train'])
                    # accs[dist].append(value['accuracies'][dist])

                    acc_total_current_num_train += value['accuracies'][dist]
                    acc_cnt += 1
            accs2[dist].append(acc_total_current_num_train / acc_cnt)

    accs_at_25k = [] 
    for _, value in accs2.items():
        accs_at_25k.append(value[-1])
    sort_idx = np.flip(np.argsort(accs_at_25k))
    sorted_dists = np.array(test_dists)[sort_idx].tolist()

    palette = sns.color_palette('RdBu', n_colors=20)
    for dist,col in zip(sorted_dists, palette):
        if dist == 'normal':
            # sns_plt = sns.lineplot(x=depths[dist], y=accs[dist], color='red')
            sns_plt = sns.lineplot(x=num_trains, y=accs2[dist], color='red')
        else:
            # sns_plt = sns.lineplot(x=depths[dist], y=accs[dist], color=col)
            sns_plt = sns.lineplot(x=num_trains, y=accs2[dist], color=col)

    # Shrink current axis by 20%
    box = sns_plt.get_position()
    sns_plt.set_position([box.x0, box.y0, box.width * 0.85, box.height])

    title = f'DenseNet: Test Acc on Distorted CIFAR10 Images With Small Training Datasets'
    # sns_plt.legend(title='Distortions', labels=test_dists, loc='center left', bbox_to_anchor=(1.18, 0.5))
    sns_plt.legend(title='Distortions', labels=sorted_dists, loc='center left', bbox_to_anchor=(1.18, 0.5))
    sns_plt.set(xlabel='Number of Training Data Images', ylabel='Test/Acc', title=title)
    print(title.replace(' ', '-').lower())
    # save_graph(plt, title.replace(' ', '-').lower())
    if show_graph:
        plt.show()
    return plt, sns_plt


def save_graph(sns_plt, path):
    # figure = sns_plt.get_figure()
    figure = sns_plt
    figure.tight_layout()
    figure.savefig(f'figs/{path}.pdf')


#%%
dist = 'normal'
num_train = 1000


#%%
compare_dists(densenet_d)

#%%
## ResNet Analytics
print('ResNet analytics')
for num_train in num_trains:
    plt, sns_plt = analyse_arch_dict(resnet_d, num_train=num_train, distortion=dist, show_graph=False, resnet_limit=110)
plt.show()


#%%
## VGG Analyics
print('VGG analytics')
for num_train in num_trains:
    analyse_arch_dict(vgg_d, num_train=num_train, distortion=dist, show_graph=False)

#%%
## DenseNet Analyics
print('DenseNet analytics')
for num_train in num_trains:
    analyse_arch_dict(densenet_d, num_train=num_train, distortion=dist, show_graph=False)


#%%
