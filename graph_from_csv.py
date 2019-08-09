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


def save_graph(sns_plt, path):
    # figure = sns_plt.get_figure()
    figure = sns_plt
    figure.tight_layout()
    figure.savefig(f'figs/{path}.pdf')


#%%
dist = 'normal'
num_train = 1000
    

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
analyse_arch_dict(densenet_d, num_train=num_train, distortion=dist)


#%%
