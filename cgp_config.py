#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

# network configurations
class CgpInfoConvSet(object):
    def __init__(self, arch_type='resnet', rows=30, cols=40, level_back=40,
                 min_active_num=8, max_active_num=50):
        self.input_num = 1
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
        if arch_type == 'resnet':
            self.func_type = func_type_resnet
        elif arch_type == 'vgg':
            self.func_type = func_type_vgg

        self.func_in_num = [1, 1,
                            1, 1,
                            1, 1,
                            1, 1,
                            1, 1,
                            1, 1,
                            2, 2,
                            1, 1]

        self.out_num = 1
        self.out_type = ['full']
        self.out_in_num = [1]

        # CGP network configuration
        self.rows = rows
        self.cols = cols
        self.node_num = rows * cols
        self.level_back = level_back
        self.min_active_num = min_active_num
        self.max_active_num = max_active_num

        self.func_type_num = len(self.func_type)
        self.out_type_num = len(self.out_type)
        self.max_in_num = np.max(
            [np.max(self.func_in_num), np.max(self.out_in_num)])
