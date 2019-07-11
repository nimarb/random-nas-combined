#!/usr/bin/env python3

import torch
import torch.nn as nn
from torch.autograd import Variable
from collections import OrderedDict
import math
import torch.nn.functional as F
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
        x = self.conv1(inputs1)
        in_data = [x, inputs2]
        # # check of the image size
        # if (in_data[0].size(2) - in_data[1].size(2)) != 0:
        #     small_in_id, large_in_id = (0, 1) if in_data[0].size(2) < in_data[1].size(2) else (1, 0)
        #     pool_num = math.floor(in_data[large_in_id].size(2) / in_data[small_in_id].size(2))
        #     for _ in range(pool_num-1):
        #         in_data[large_in_id] = F.max_pool2d(in_data[large_in_id], 2, 2, 0)

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
    def __init__(self, cgp, in_channel, n_class, img_size, arch_type='resnet'):
        super(CGP2CNN, self).__init__()
        self.cgp = cgp
        self.arch = OrderedDict()
        self.encode = []
        self.channel_num = [None for _ in range(500)]
        self.size = [None for _ in range(500)]
        self.channel_num[0] = in_channel
        self.size[0] = img_size
        # encoder
        i = 0
        if arch_type == 'resnet':
            for name, in1, in2 in self.cgp:
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

    def main(self, x):
        outputs = self.outputs
        outputs[0] = x    # input image
        nodeID = 1
        for layer in self.layer_module:
            if isinstance(layer, SepConv):
                outputs[nodeID] = layer(outputs[self.cgp[nodeID][1]])
            elif isinstance(layer, DilConv):
                outputs[nodeID] = layer(outputs[self.cgp[nodeID][1]])
            elif isinstance(layer, ResBlock):
                outputs[nodeID] = layer(outputs[self.cgp[nodeID][1]],
                                        outputs[self.cgp[nodeID][1]])
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
