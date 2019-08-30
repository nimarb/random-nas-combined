#!/usr/bin/env python3

import torch
import torch.nn as nn
from torch.autograd import Variable
from collections import OrderedDict
import math
from math import sqrt
from utils import cov
import torch.nn.functional as F
import torch.utils.checkpoint as cp
import sys


class ConvBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel, stride):
        super(ConvBlock, self).__init__()
        pad_size = kernel // 2
        self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, kernel,
                                             stride=stride, padding=pad_size,
                                             bias=False),
                                   nn.BatchNorm2d(out_size),
                                   nn.ReLU(inplace=True),)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        return outputs


class DeConvBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel):
        super(DeConvBlock, self).__init__()
        pad_size = kernel // 2
        self.conv1 = nn.Sequential(nn.ConvTranspose2d(in_size, out_size,
                                                      kernel, stride=2,
                                                      padding=pad_size,
                                                      output_padding=1,
                                                      bias=False),
                                   nn.BatchNorm2d(out_size),
                                   nn.ReLU(inplace=True),)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        return outputs


class ConvBlock_last(nn.Module):
    def __init__(self, in_size, out_size, kernel):
        super(ConvBlock_last, self).__init__()
        pad_size = kernel // 2
        self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, kernel,
                                             padding=pad_size, bias=False))
        # nn.BatchNorm2d(out_size),
        # nn.Tanh())

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        return outputs


class DeConvBlock_last(nn.Module):
    def __init__(self, in_size, out_size, kernel):
        super(DeConvBlock_last, self).__init__()
        pad_size = kernel // 2
        self.conv1 = nn.Sequential(nn.ConvTranspose2d(in_size, out_size,
                                                      kernel, padding=pad_size,
                                                      bias=False))
        # nn.BatchNorm2d(out_size),
        # nn.Tanh())

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        return outputs


class ConvBlock_s(nn.Module):
    def __init__(self, in_size, out_size, kernel, stride):
        super(ConvBlock_s, self).__init__()
        pad_size = kernel // 2
        self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, kernel,
                                             stride=stride, padding=pad_size,
                                             bias=False),
                                   nn.BatchNorm2d(out_size),
                                   nn.ReLU(inplace=True),)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        return outputs


class ConvBlock_sum(nn.Module):
    def __init__(self, in_size, out_size, kernel):
        super(ConvBlock_sum, self).__init__()
        pad_size = kernel // 2
        self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, kernel,
                                             padding=pad_size, bias=False),
                                   nn.BatchNorm2d(out_size),
                                   nn.ReLU(inplace=True),)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2):
        outputs = self.conv1(inputs1)
        in_data = [outputs, inputs2]
        # check of the channel size
        if in_data[0].size(1) < in_data[1].size(1):
            small_ch_id, large_ch_id = (0, 1)
        else:
            small_ch_id, large_ch_id = (1, 0)
        offset = int(in_data[large_ch_id].size()[1]
                     - in_data[small_ch_id].size()[1])
        if offset != 0:
            tmp = in_data[large_ch_id].data[:, :offset, :, :]
            tmp = Variable(tmp).clone()
            in_data[small_ch_id] = torch.cat(
                [in_data[small_ch_id], tmp * 0], 1)
        out = torch.add(in_data[0], in_data[1])
        return self.relu(out)


class ResBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel, stride):
        super(ResBlock, self).__init__()
        pad_size = kernel // 2
        self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, kernel,
                                             stride=stride, padding=pad_size,
                                             bias=False),
                                   nn.BatchNorm2d(out_size),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(out_size, out_size, kernel,
                                             stride=stride, padding=pad_size,
                                             bias=False),
                                   nn.BatchNorm2d(out_size))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2):
        # inputs1, inputs2 is a batch of image data
        x = self.conv1(inputs1)
        in_data = [x, inputs2]
        # # check of the image size
        # if (in_data[0].size(2) - in_data[1].size(2)) != 0:
        #     small_in_id, large_in_id = (0, 1) if in_data[0].size(2) < in_data[1].size(2) else (1, 0)
        #     pool_num = math.floor(in_data[large_in_id].size(2) / in_data[small_in_id].size(2))
        #     for _ in range(pool_num-1):
        #         in_data[large_in_id] = F.max_pool2d(in_data[large_in_id], 2, 2, 0)

        # check of the channel size
        # torch.Size([128, 64, 32, 32]) --> size(1) = 64
        if in_data[0].size(1) < in_data[1].size(1):
            small_ch_id, large_ch_id = (0, 1)
        else:
            small_ch_id, large_ch_id = (1, 0)
        offset = int(in_data[large_ch_id].size()[1]
                     - in_data[small_ch_id].size()[1])
        if offset != 0:
            # This piece of codes enlarges the smaller tensor so as to match
            # the size of the bigger tensor by padding it with zeros. This is in
            # the channel (filter) dimension.
            tmp = in_data[large_ch_id].data[:, :offset, :, :]
            tmp = Variable(tmp).clone()
            in_data[small_ch_id] = torch.cat(
                [in_data[small_ch_id], tmp * 0], 1)
        out = torch.add(in_data[0], in_data[1])
        return self.relu(out)


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate,
                 memory_efficient=False, kernel_size=3):
        super(_DenseLayer, self).__init__()
        pad_size = (kernel_size - 1) // 2
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=kernel_size, stride=1, padding=pad_size,
                                           bias=False)),
        self.drop_rate = drop_rate
        self.memory_efficient = memory_efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.memory_efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features


def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        ##########
        # Debug
        if concated_features.shape[1] != conv.in_channels:
            print('Features vs conv size mismatch')
        ##########
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function


class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate,
                 drop_rate, memory_efficient=False, kernel_size=3):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
                kernel_size=kernel_size
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        """The num_output_features has to be the same as the number of input
        features (growth_rate) as the next DenseBlock"""
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))



class DenseBlockTorch(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
                            is also the number of output channels
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    # def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                #  num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000, memory_efficient=False):
    def __init__(self, num_init_features, growth_rate, num_layers, kernel_size=3,
                 bn_size=4, drop_rate=0, memory_efficient=True, is_first=False,
                 is_last=False, num_trans_out=None):
        # in, out, kernel, stride

        super(DenseBlockTorch, self).__init__()

        # First convolutions if applicable
        self.is_first = is_first
        if self.is_first:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,
                                    padding=3, bias=False)),
                ('norm0', nn.BatchNorm2d(num_init_features)),
                ('relu0', nn.ReLU(inplace=True)),
                ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
            ]))
        else:
            self.features = nn.Sequential()

        # Each denseblock
        num_features = num_init_features
        # for i, num_layers in enumerate(block_config):
        i = 0
        block = _DenseBlock(
            num_layers=num_layers,
            num_input_features=num_features,
            bn_size=bn_size,
            growth_rate=growth_rate,
            drop_rate=drop_rate,
            memory_efficient=memory_efficient,
            kernel_size=kernel_size
        )
        self.features.add_module('denseblock%d' % (i + 1), block)
        num_features = num_features + num_layers * growth_rate

        self.is_last = is_last
        if not self.is_last:
        # if i != len(block_config) - 1:
            if None == num_trans_out:
                num_trans_out = growth_rate
            trans = _Transition(num_input_features=num_features,
                                num_output_features=num_trans_out)
                                # num_output_features=num_features // 2)
            self.features.add_module('transition%d' % (i + 1), trans)
            num_features = num_features // 2
            # num_features = growth_rate
        else:
            self.num_last_features = num_features
            self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        # if self.is_last:
            # out = F.relu(features, inplace=True)
            # out = F.adaptive_avg_pool2d(out, (1, 1))
        return features


class DenseFinal(nn.Module):
    def __init__(self):
        super(DenseFinal, self).__init__()

        # Final batch norm
        num_features = 64
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        # Linear layer
        num_classes = 10
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        out = F.relu(x, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = self.classifier(out)
        return out


class Sum(nn.Module):
    def __init__(self):
        super(Sum, self).__init__()

    def forward(self, inputs1, inputs2):
        in_data = [inputs1, inputs2]
        # check of the image size
        if (in_data[0].size(2) - in_data[1].size(2)) != 0:
            if in_data[0].size(2) < in_data[1].size(2):
                small_in_id, large_in_id = (0, 1)
            else:
                small_in_id, large_in_id = (1, 0)
            pool_num = int(math.log2(in_data[large_in_id].size(2))
                           - math.log2(in_data[small_in_id].size(2)))
            for _ in range(pool_num):
                in_data[large_in_id] = F.max_pool2d(
                    in_data[large_in_id], 2, 2, 0)
        # check of the channel size
        if in_data[0].size(1) < in_data[1].size(1):
            small_ch_id, large_ch_id = (0, 1)
        else:
            small_ch_id, large_ch_id = (1, 0)
        offset = int(in_data[large_ch_id].size()[1]
                     - in_data[small_ch_id].size()[1])
        if offset != 0:
            tmp = in_data[large_ch_id].data[:, :offset, :, :]
            tmp = Variable(tmp).clone()
            in_data[small_ch_id] = torch.cat(
                [in_data[small_ch_id], tmp * 0], 1)
        out = torch.add(in_data[0], in_data[1])
        return out


class Concat(nn.Module):
    def __init__(self):
        super(Concat, self).__init__()

    def forward(self, inputs1, inputs2):
        in_data = [inputs1, inputs2]
        # check of the image size
        if (in_data[0].size(2) - in_data[1].size(2)) != 0:
            if in_data[0].size(2) < in_data[1].size(2):
                small_in_id, large_in_id = (0, 1)
            else:
                small_in_id, large_in_id = (1, 0)
            pool_num = int(math.log2(in_data[large_in_id].size(2))
                           - math.log2(in_data[small_in_id].size(2)))
            for _ in range(pool_num):
                in_data[large_in_id] = F.max_pool2d(in_data[large_in_id], 2, 2,
                                                    0)
        return torch.cat([in_data[0], in_data[1]], 1)


class DeConvBlock_sum(nn.Module):
    def __init__(self, in_size, out_size, kernel):
        super(DeConvBlock_sum, self).__init__()
        pad_size = kernel // 2
        self.conv1 = nn.Sequential(nn.ConvTranspose2d(in_size, out_size,
                                                      kernel, stride=2,
                                                      padding=pad_size,
                                                      output_padding=1,
                                                      bias=False),
                                   nn.BatchNorm2d(out_size),
                                   nn.ReLU(inplace=True),)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2):
        outputs1 = self.conv1(inputs1)
        offset = outputs1.size()[2] - inputs2.size()[2]
        padding = 2 * [offset // 2, offset // 2]
        outputs2 = F.pad(inputs2, padding)
        out = torch.add(outputs1, outputs2)
        return self.relu(out)


class SepConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride=1, affine=False):
        super(SepConv, self).__init__()
        pad_size = kernel_size // 2
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride,
                      padding=pad_size, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1,
                      padding=pad_size, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),)

    def forward(self, x):
        return self.op(x)


class DilConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, dilation=2, stride=1,
                 affine=False):
        super(DilConv, self).__init__()
        pad_size = kernel_size - dilation + 1
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride,
                      padding=pad_size, dilation=dilation, groups=C_in,
                      bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),)

    def forward(self, x):
        return self.op(x)


class CGP2CNN(nn.Module):
    def __init__(self, cgp, in_channel, n_class, img_size, arch_type='resnet',
                 register_hook=True, num_layer_eig=3, layer_eig_spacing=1):
        super(CGP2CNN, self).__init__()
        self.cgp = cgp
        self.arch = OrderedDict()
        self.encode = []
        self.channel_num = [None for _ in range(500)]
        self.size = [None for _ in range(500)]
        self.channel_num[0] = in_channel
        self.size[0] = img_size
        self.densenet_is_first = False
        # self.layer_channels = {}
        self.covariance_matrices = []
        self.eigenvalues = []
        self.register_hook = register_hook
        # encoder
        i = 0
        if arch_type == 'resnet':
            for name, in1, in2 in self.cgp:
            # for name, in1, in self.cgp:
                # in2 = in1
                if name == 'input' in name:
                    i += 1
                    continue
                elif name == 'full':
                    self.encode.append(nn.Linear(self.channel_num[in1],
                                                 n_class))
                elif name == 'Max_Pool' or name == 'Avg_Pool':
                    self.channel_num[i] = self.channel_num[in1]
                    self.size[i] = int(self.size[in1] / 2)
                    key = name.split('_')
                    func = key[0]
                    if func == 'Max':
                        self.encode.append(nn.MaxPool2d(2, 2))
                    else:
                        self.encode.append(nn.AvgPool2d(2, 2))
                elif name == 'Concat':
                    self.channel_num[i] = self.channel_num[in1] \
                        + self.channel_num[in2]
                    if self.size[in1] < self.size[in2]:
                        small_in_id, large_in_id = (in1, in2)
                    else:
                        small_in_id, large_in_id = (in2, in1)
                    self.size[i] = self.size[small_in_id]
                    self.encode.append(Concat())
                elif name == 'Sum':
                    if self.channel_num[in1] < self.channel_num[in2]:
                        small_in_id, large_in_id = (in1, in2)
                    else:
                        small_in_id, large_in_id = (in2, in1)
                    self.channel_num[i] = self.channel_num[large_in_id]
                    if self.size[in1] < self.size[in2]:
                        small_in_id, large_in_id = (in1, in2)
                    else:
                        small_in_id, large_in_id = (in2, in1)
                    self.size[i] = self.size[small_in_id]
                    self.encode.append(Sum())
                else:
                    key = name.split('_')
                    down = key[0]
                    func = key[1]
                    out_size = int(key[2])
                    kernel = int(key[3])
                    if down == 'S':
                        if func == 'SepBlock':
                            self.channel_num[i] = out_size
                            self.size[i] = self.size[in1]
                            self.encode.append(SepConv(self.channel_num[in1],
                                                       out_size, kernel))
                        elif func == 'DilBlock':
                            self.channel_num[i] = out_size
                            self.size[i] = self.size[in1]
                            self.encode.append(DilConv(self.channel_num[in1],
                                                       out_size, kernel))
                        elif func == 'ResBlock':
                            in_data = [out_size, self.channel_num[in1]]
                            if in_data[0] < in_data[1]:
                                small_in_id, large_in_id = (0, 1)
                            else:
                                small_in_id, large_in_id = (1, 0)
                            self.channel_num[i] = in_data[large_in_id]
                            self.size[i] = self.size[in1]
                            self.encode.append(ResBlock(self.channel_num[in1],
                                                        out_size, kernel,
                                                        stride=1))
                        else:
                            sys.exit("error at CGPCNN init")
                    else:
                        sys.exit('error at CGPCNN init')
                i += 1

        elif arch_type == 'densenet':
             for idx, (name, in1) in enumerate(self.cgp):
                if name == 'input' in name:
                    self.densenet_is_first = True
                    i += 1
                    continue
                elif name == 'full':
                    self.encode.append(nn.Linear(self.num_last_features,
                                                 n_class))
                    # self.encode.append(nn.Linear(self.channel_num[in1],
                                                #  n_class))
                elif name == 'Max_Pool' or name == 'Avg_Pool':
                    self.channel_num[i] = self.channel_num[in1]
                    self.size[i] = int(self.size[in1] / 2)
                    key = name.split('_')
                    func = key[0]
                    if func == 'Max':
                        self.encode.append(nn.MaxPool2d(2, 2))
                    else:
                        self.encode.append(nn.AvgPool2d(2, 2))
                else:
                    key = name.split('_')
                    down = key[0]
                    func = key[1]
                    out_size = int(key[2])
                    kernel = int(key[3])
                    real_kernel = int(key[4])
                    if down == 'S':
                        if func == 'SepBlock':
                            self.channel_num[i] = out_size
                            self.size[i] = self.size[in1]
                            self.encode.append(SepConv(self.channel_num[in1],
                                                       out_size, kernel))
                        elif func == 'DilBlock':
                            self.channel_num[i] = out_size
                            self.size[i] = self.size[in1]
                            self.encode.append(DilConv(self.channel_num[in1],
                                                       out_size, kernel))
                        elif func == 'DenseBlock':
                            in_data = [out_size, self.channel_num[in1]]
                            if in_data[0] < in_data[1]:
                                small_in_id, large_in_id = (0, 1)
                            else:
                                small_in_id, large_in_id = (1, 0)
                            self.channel_num[i] = in_data[large_in_id]
                            self.size[i] = self.size[in1]
                            # out_size = [16, 32, 64], kernel = [6, 12, 24, 16]
                            # kernel is the number of layers per block
                            # out_size is the growth rate
                            # in_size should be the prev output
                            is_last = False
                            if idx == len(self.cgp) - 2:
                                is_last = True
                                in_size_next = None
                            else:
                                name_next, _ = self.cgp[i+1]
                                key_next = name_next.split('_')
                                in_size_next = int(key_next[2])
                                in_size_next = self.channel_num[in1+1]
                            self.encode.append(DenseBlockTorch(self.channel_num[in1],
                                                          out_size, kernel,
                                                          kernel_size=real_kernel,
                                                          is_first=self.densenet_is_first,
                                                          is_last=is_last,
                                                          num_trans_out=in_size_next))
                            if is_last:
                                self.num_last_features = self.encode[-1].num_last_features
                            self.densenet_is_first = False
                        else:
                            sys.exit("error at CGPCNN init")
                    else:
                        sys.exit('error at CGPCNN init')
                i += 1

        elif arch_type == 'vgg':
            for name, in1 in self.cgp:
                if name == 'input' in name:
                    i += 1
                    continue
                elif name == 'full':
                    self.encode.append(nn.Linear(self.channel_num[in1],
                                                 n_class))
                elif name == 'Max_Pool' or name == 'Avg_Pool':
                    self.channel_num[i] = self.channel_num[in1]
                    self.size[i] = int(self.size[in1] / 2)
                    key = name.split('_')
                    func = key[0]
                    if func == 'Max':
                        self.encode.append(nn.MaxPool2d(2, 2))
                    else:
                        self.encode.append(nn.AvgPool2d(2, 2))
                else:
                    key = name.split('_')
                    down = key[0]
                    func = key[1]
                    out_size = int(key[2])
                    kernel = int(key[3])
                    if down == 'S':
                        if func == 'SepBlock':
                            self.channel_num[i] = out_size
                            self.size[i] = self.size[in1]
                            self.encode.append(SepConv(self.channel_num[in1],
                                                       out_size, kernel))
                        elif func == 'DilBlock':
                            self.channel_num[i] = out_size
                            self.size[i] = self.size[in1]
                            self.encode.append(DilConv(self.channel_num[in1],
                                                       out_size, kernel))
                        elif func == 'ResBlock':
                            in_data = [out_size, self.channel_num[in1]]
                            if in_data[0] < in_data[1]:
                                small_in_id, large_in_id = (0, 1)
                            else:
                                small_in_id, large_in_id = (1, 0)
                            self.channel_num[i] = in_data[large_in_id]
                            self.size[i] = self.size[in1]
                            self.encode.append(ResBlock(self.channel_num[in1],
                                                        out_size, kernel,
                                                        stride=1))
                        else:
                            sys.exit("error at CGPCNN init")
                    else:
                        sys.exit('error at CGPCNN init')
                i += 1

        self.layer_module = nn.ModuleList(self.encode)
        self.outputs = [None for _ in range(len(self.cgp))]

        layers_to_reg = []

        if self.register_hook:
            if arch_type == 'vgg' or arch_type == 'resnet':
                for name, layer in self.layer_module._modules.items():
                    if isinstance(layer, SepConv) or \
                        isinstance(layer, DilConv):
                        for actual_layer in layer.op:
                            if isinstance(actual_layer, nn.Conv2d):
                                layers_to_reg.append(actual_layer)

            elif arch_type == 'densenet':
                for name, layer in self.layer_module._modules.items():
                    if isinstance(layer, SepConv) or isinstance(layer, DilConv) or \
                    isinstance(layer,ResBlock) or isinstance(layer,DenseBlockTorch):
                        for actual_layer in layer.features:
                            if isinstance(actual_layer, nn.Conv2d):
                                layers_to_reg.append(actual_layer)
                            if isinstance(actual_layer, _DenseBlock):
                                for _, dl in actual_layer._modules.items():
                                    if isinstance(dl, nn.Conv2d):
                                        layers_to_reg.append(dl)
                            if isinstance(actual_layer, _Transition):
                                for _, dl in actual_layer._modules.items():
                                    if isinstance(dl, nn.Conv2d):
                                        layers_to_reg.append(dl)
                                    

            # elif arch_type == 'resnet':
            #     for name, layer in self.layer_module._modules.items():
            #         if isinstance(layer, SepConv) or isinstance(layer, DilConv) or \
            #         isinstance(layer,ResBlock) or isinstance(layer,DenseBlockTorch):
            #             for _, actual_layer in layer._modules['op']._modules.items():
            #                 if isinstance(actual_layer, nn.Conv2d):
            #                     layer_to_reg = actual_layer

            actual_layers_to_reg = []
            for idx, layer in enumerate(reversed(layers_to_reg)):
                if 0 == idx:
                    actual_layers_to_reg.append(layer)
                elif idx % layer_eig_spacing == 0:
                    actual_layers_to_reg.append(layer)
                if num_layer_eig <= len(actual_layers_to_reg):
                    break

            for layer in actual_layers_to_reg:
                layer.register_forward_hook(self.hook_fn)


    def hook_fn(self, module, input, output):
        # self.layer_channels[module] = output  # here not needed b/c not used again
        # pool = nn.AdaptiveAvgPool2d(10)
        pool = nn.AvgPool2d(output.size()[2:])
        analyse = pool(output)
        analyse = analyse.view(analyse.size()[0], -1, 1)
        if 0 < len(analyse):
            analyse = torch.cat((analyse[0], analyse[1]), dim=1)
        covm = cov(analyse)
        self.covariance_matrices.append(covm)
        self.eigenvalues.append(torch.symeig(covm))
        # print(covm)
        # print(torch.eig(covm))

        # size2d = int(sqrt(analyse.size()[0]))
        # covm = self.cov_complex(analyse[0,:,:,:].view(-1, size2d), analyse[0,:,:,:].view(-1, size2d))
        # print(self.layer_channels)

    def main(self, x):
        outputs = self.outputs
        outputs[0] = x    # input image
        nodeID = 1
        # print(self.layer_module)
        for idx, layer in enumerate(self.layer_module):
            if isinstance(layer, SepConv):
                outputs[nodeID] = layer(outputs[self.cgp[nodeID][1]])
            elif isinstance(layer, DilConv):
                outputs[nodeID] = layer(outputs[self.cgp[nodeID][1]])
            elif isinstance(layer, ResBlock):
                outputs[nodeID] = layer(outputs[self.cgp[nodeID][1]],
                                        outputs[self.cgp[nodeID][1]])
            elif isinstance(layer, DenseBlockTorch):
                outputs[nodeID] = layer(outputs[self.cgp[nodeID][1]])
            elif isinstance(layer, torch.nn.modules.pooling.MaxPool2d) \
                    or isinstance(layer, torch.nn.modules.pooling.AvgPool2d):
                if outputs[self.cgp[nodeID][1]].size(2) > 1:
                    outputs[nodeID] = layer(outputs[self.cgp[nodeID][1]])
                else:
                    outputs[nodeID] = outputs[self.cgp[nodeID][1]]
            elif isinstance(layer, Concat):
                outputs[nodeID] = layer(outputs[self.cgp[nodeID][1]],
                                        outputs[self.cgp[nodeID][2]])
            elif isinstance(layer, Sum):
                outputs[nodeID] = layer(outputs[self.cgp[nodeID][1]],
                                        outputs[self.cgp[nodeID][2]])
            elif isinstance(layer, torch.nn.modules.linear.Linear):
                tmp = F.adaptive_avg_pool2d(outputs[self.cgp[nodeID][1]], 1)
                tmp = tmp.view(tmp.size(0), -1)
                outputs[nodeID] = layer(tmp)
            else:
                print(layer)
                sys.exit("Error at CGP2CNN forward")
            nodeID += 1
        return outputs[nodeID-1]

    def forward(self, x):
        return self.main(x)
