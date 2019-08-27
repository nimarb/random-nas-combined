import argparse
import csv
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default='save_dir')
    parser.add_argument('--arch_type', type=str, default='densenet')
    parser.add_argument('--num_trains', type=int, nargs='+', default=[5000, 10000])
    args = parser.parse_args()
    return args


def get_nr_conv_layers(layer_name):
    """Returns the number of convolution layers contained in a specific layer
    type defined in the cgp_config"""

    # RESNET START
    layer_types = ['S_SepBlock_16_3',  'S_SepBlock_16_5',
                        'S_SepBlock_32_3',  'S_SepBlock_32_5',
                        'S_SepBlock_64_3',  'S_SepBlock_64_5',
                        'S_ResBlock_16_3',  'S_ResBlock_16_5',
                        'S_ResBlock_32_3',  'S_ResBlock_32_5',
                        'S_ResBlock_64_3',  'S_ResBlock_64_5',
                        'Sum','Sum',
                        'Max_Pool', 'Avg_Pool',
                        # VGG blocks dup of resnet
                        # DENSE START
                        'S_DenseBlock_16_3_3',  'S_DenseBlock_16_6_3',
                        'S_DenseBlock_16_3_5',  'S_DenseBlock_16_6_5',
                        'S_DenseBlock_16_12_3',  'S_DenseBlock_16_9_3',
                        'S_DenseBlock_16_12_5',  'S_DenseBlock_16_9_5',
                        'S_DenseBlock_32_6_3',  'S_DenseBlock_32_12_3',
                        'S_DenseBlock_32_6_5',  'S_DenseBlock_32_12_5',
                        'S_DenseBlock_32_3_3',  'S_DenseBlock_32_9_3',
                        'S_DenseBlock_32_3_5',  'S_DenseBlock_32_9_5',
                        'S_DenseBlock_64_6_3',  'S_DenseBlock_64_12_3',
                        'S_DenseBlock_64_6_5',  'S_DenseBlock_64_12_5',
                        'S_DenseBlock_64_3_3',  'S_DenseBlock_64_9_3',
                        'S_DenseBlock_64_3_5',  'S_DenseBlock_64_9_5']

    num_conv_layers = 0
    if 'SepBlock' in layer_name:
        num_conv_layers = 4
    elif 'DilConv' in layer_name:
        num_conv_layers = 2
    elif 'DenseBlock' in layer_name:
        feats = layer_name.split('_')
        dense_layer_num = int(feats[3])
        num_conv_layers = dense_layer_num * 2
    elif 'ResBlock' in layer_name:
        num_conv_layers = 2
    elif 'ConvBlock' in layer_name:
        num_conv_layers = 1
    else:
        num_conv_layers = 0

    return num_conv_layers


def get_arch_dict(logdir='save_dir', net_type='resnet', base_date='2019-07'):
    """ Looks like:
        arch_dict[0..num_trials]
        arch_dict[0]['accuracies']['brightness'] = 0.2134234
        arch_dict[0]['num_depth'] = 47"""
    if isinstance(logdir, str):
        logdir = Path(logdir)

    base_str = f'{net_type}-{base_date}-*-*'
    acc_paths = sorted(logdir.glob(f'{base_str}/accuracies.json'))
    arch_paths = sorted(logdir.glob(f'{base_str}/log-active.txt'))

    arch_dict = {}
    accs = []
    archs = []

    # combine accuracies and network architecture
    acc_ctr = 0
    for acc_p in acc_paths:
        with open(acc_p, 'r') as acc_f:
            acc = json.load(acc_f)
            for _, value in acc.items():
                with open(acc_p.parent.joinpath('config.json')) as cfg_f:
                    config = json.load(cfg_f)
                arch_dict[acc_ctr] = config
                arch_dict[acc_ctr]['accuracies'] = value
                acc_ctr += 1

    # extract actual layer depth and arch string from `log-active.txt` file
    arch_ctr = 0
    for arch_p in arch_paths:
        with open(arch_p, 'r') as arch_f:
            lines = arch_f.readlines()
            for line in lines:
                idx = line.index(',,')
                idx_depth_end = line.index(',"')
                arch_str = line[idx_depth_end+2:-2]
                num_depth = line[idx+2:idx_depth_end]
                arch_dict[arch_ctr]['num_depth'] = int(num_depth)
                arch_dict[arch_ctr]['arch_str'] = arch_str
                # count the actual number of conv layers
                arch_list_raw = arch_str.split(',')
                arch_list = [i for i in arch_list_raw if i[1] == "["]
                num_conv_layers = 0
                for entry in arch_list:
                    num_conv_layers += get_nr_conv_layers(entry)
                if 'densenet' == net_type:
                    num_conv_layers += 1
                arch_dict[arch_ctr]['num_conv_layers'] = num_conv_layers

                arch_ctr += 1


    # arch_dict['num_train'] = num_train
    # arch_dict['avg_acc'] = sum(accs) / len(accs)
    # arch_dict['max_acc'] = max(accs)
    # arch_dict['archs'] = archs
    return arch_dict


if __name__ == "__main__":
    args = parse_args()
    arch_type = args.arch_type
    arch_d = get_arch_dict(net_type=arch_type, base_date=2019)
    with open(f'{arch_type}.json', 'w+') as fout:
        json.dump(arch_d, fout, indent=2)
    print(f'Saved the consolidated results json to {arch_type}.json')
