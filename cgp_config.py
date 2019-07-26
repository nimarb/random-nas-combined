#!/usr/bin/env python3

import numpy as np


class CgpInfoConvSet(object):
    """Network configurations"""
    def __init__(self, arch_type='resnet', rows=30, cols=40, level_back=40,
                 min_active_num=8, max_active_num=50):
        self.input_num = 1  # number (ID) of the input node (usually 1 (start))
        func_type_resnet = ['S_SepBlock_16_3',  'S_SepBlock_16_5',
                            'S_SepBlock_32_3',  'S_SepBlock_32_5',
                            'S_SepBlock_64_3',  'S_SepBlock_64_5',
                            'S_ResBlock_16_3',  'S_ResBlock_16_5',
                            'S_ResBlock_32_3',  'S_ResBlock_32_5',
                            'S_ResBlock_64_3',  'S_ResBlock_64_5',
                            'Sum','Sum',
                            'Max_Pool', 'Avg_Pool']
        func_type_vgg = ['S_SepBlock_16_3',  'S_SepBlock_16_5',
                         'S_SepBlock_32_3',  'S_SepBlock_32_5',
                         'S_SepBlock_64_3',  'S_SepBlock_64_5',
                         'Max_Pool', 'Avg_Pool']
        func_type_densenet = ['S_SepBlock_16_3',  'S_SepBlock_16_5',
                            'S_SepBlock_32_3',  'S_SepBlock_32_5',
                            'S_SepBlock_64_3',  'S_SepBlock_64_5',
                            'S_DenseBlock_16_3',  'S_DenseBlock_16_5',
                            'S_DenseBlock_32_3',  'S_DenseBlock_32_5',
                            'S_DenseBlock_64_3',  'S_DenseBlock_64_5',
                            'Sum','Sum',
                            'Max_Pool', 'Avg_Pool']
        
        func_in_num_resnet = [1, 1,
                              1, 1,
                              1, 1,
                              1, 1,
                              1, 1,
                              1, 1,
                              2, 2,
                              1, 1]
        func_in_num_vgg = [1, 1,
                           1, 1,
                           1, 1,
                           1, 1,
                           1, 1,
                           1, 1,
                           1, 1]
        func_in_num_densenet = [1, 1,
                              1, 1,
                              1, 1,
                              1, 1,
                              1, 1,
                              1, 1,
                              2, 2,
                              1, 1]

        if arch_type == 'resnet':
            self.func_type = func_type_resnet
            self.func_in_num = func_in_num_resnet
        elif arch_type == 'vgg':
            self.func_type = func_type_vgg
            self.func_in_num = func_in_num_vgg
        elif arch_type == 'densenet':
            self.func_type = func_type_densenet
            self.func_in_num = func_in_num_densenet

        self.out_num = 1
        self.out_type = ['full']
        self.out_in_num = [1]  # 

        # CGP network configuration
        self.rows = rows  # For most NNs `= 1` to make a simple NN
        self.cols = cols  # Actual number of layers if `cols = 1`
        self.node_num = rows * cols  # Number of network layers
        self.level_back = level_back  # Used for mod of connection gene in cgp
        self.min_active_num = min_active_num
        self.max_active_num = max_active_num

        self.func_type_num = len(self.func_type)
        self.out_type_num = len(self.out_type)
        self.max_in_num = np.max(
            [np.max(self.func_in_num), np.max(self.out_in_num)])
