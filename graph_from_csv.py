#%%

import numpy as np
import csv
import json
import argparse
import seaborn as sns
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt

#%%
with open('resnet.json', 'r') as fin:
    resnet_d = json.load(fin)
with open('vgg.json', 'r') as fin:
    vgg_d = json.load(fin)

num_trains = [500, 1000, 5000, 10000, 25000]

def analyse_arch_dict(arch_dict, num_train=1000):
    depths = []
    normal_accs = []
    for _, value in arch_dict.items():
        if value['num_train'] == num_train:
            depths.append(value['num_depth'])
            normal_accs.append(value['accuracies']['normal'])

    sns.lineplot(x=(range(len(depths))), y=depths)
    plt.show()
    print(f'max val: {max(depths)}')
    print(f'min val: {min(depths)}')
    print(f'avg val: {sum(depths)/len(depths)}')
    sorted_depth = sorted(depths)
    sns.lineplot(x=depths, y=normal_accs)
    plt.show()
    

#%%
## ResNet Analytics
print('ResNet analytics')
analyse_arch_dict(resnet_d, num_train=5000)

#%%
## VGG Analyics
print('VGG analytics')
analyse_arch_dict(vgg_d, num_train=25000)
