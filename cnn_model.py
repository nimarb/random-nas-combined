#!/usr/bin/env python3

import torch
import torch.nn as nn
from torch.autograd import Variable
from collections import OrderedDict
import math
from math import sqrt
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
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
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
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function


class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
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
    def __init__(self, num_init_features, growth_rate, num_layers,
                 bn_size=4, drop_rate=0, memory_efficient=True):
        # in, out, kernel, stride

        super(DenseBlockTorch, self).__init__()
        self.features = nn.Sequential()
        # Each denseblock
        num_features = num_init_features
        # for i, num_layers in enumerate(block_config):
        i = 0
        block_config = []
        block = _DenseBlock(
            num_layers=num_layers,
            num_input_features=num_features,
            bn_size=bn_size,
            growth_rate=growth_rate,
            drop_rate=drop_rate,
            memory_efficient=memory_efficient
        )
        self.features.add_module('denseblock%d' % (i + 1), block)
        num_features = num_features + num_layers * growth_rate
        if i != len(block_config) - 1:
            trans = _Transition(num_input_features=num_features,
                                num_output_features=growth_rate)
                                # num_output_features=num_features // 2)
            self.features.add_module('transition%d' % (i + 1), trans)
            # num_features = num_features // 2
            num_features = growth_rate

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

########################################################################
# From https://raw.githubusercontent.com/kevinzakka/densenet/master/layers.py

class SubBlock(nn.Module):
    """
    This piece of the DenseBlock receives an input feature map
    x and transforms it through a dense, composite function H(x).

    The transformation H(x) is a composition of 3 consecutive 
    operations: BN - ReLU - Conv (3x3).

    In the bottleneck variant of the SubBlock, a 1x1 conv is
    added to the transformation function H(x), reducing the number
    of input feature maps and improving computational efficiency.
    """
    def __init__(self, in_channels, out_channels, bottleneck, p):
        """
        Initialize the different parts of the SubBlock.

        Params
        ------
        - in_channels: number of input channels in the convolution.
        - out_channels: number of output channels in the convolution.
        - bottleneck: if true, applies the bottleneck variant of H(x).
        - p: if greater than 0, applies dropout after the convolution.
        """
        super(SubBlock, self).__init__()
        self.bottleneck = bottleneck
        self.p = p

        in_channels_2 = in_channels
        out_channels_2 = out_channels

        if bottleneck:
            in_channels_1 = in_channels
            out_channels_1 = out_channels * 4
            in_channels_2 = out_channels_1

            self.bn1 = nn.BatchNorm2d(in_channels_1)
            self.conv1 = nn.Conv2d(in_channels_1,
                                   out_channels_1,
                                   kernel_size=1)

        self.bn2 = nn.BatchNorm2d(in_channels_2)
        self.conv2 = nn.Conv2d(in_channels_2, 
                               out_channels_2, 
                               kernel_size=3, 
                               padding=1)

    def forward(self, x):
        """
        Compute the forward pass of the composite transformation H(x),
        where x is the concatenation of the current and all preceding
        feature maps.
        """
        if self.bottleneck:
            out = self.conv1(F.relu(self.bn1(x)))
            if self.p > 0:
                out = F.dropout(out, p=self.p, training=self.training)
            out = self.conv2(F.relu(self.bn2(out)))
            if self.p > 0:
                out = F.dropout(out, p=self.p, training=self.training)
        else:
            out = self.conv2(F.relu(self.bn2(x)))
            if self.p > 0:
                out = F.dropout(out, p=self.p, training=self.training)  
        return torch.cat((x, out), 1)


class DenseBlock(nn.Module):
    """
    Block that connects L layers directly with each other in a 
    feed-forward fashion.

    Concretely, this block is composed of L SubBlocks sharing a 
    common growth rate k (Figure 1 in the paper).
    """
    def __init__(self, num_layers, in_channels, growth_rate, bottleneck, p):
        # in, out, kernel, stride
        # num_init_features, growth_rate, num_layers
        """
        Initialize the different parts of the DenseBlock.

        Params
        ------
        - num_layers: the number of layers L in the dense block.
        - in_channels: the number of input channels feeding into the first 
          subblock.
        - growth_rate: the number of output feature maps produced by each subblock.
          This number is common across all subblocks.
        """
        super(DenseBlock, self).__init__()

        # create L subblocks
        layers = []
        for i in range(num_layers):
            cumul_channels = in_channels + i * growth_rate
            layers.append(SubBlock(cumul_channels, growth_rate, bottleneck, p))

        self.block = nn.Sequential(*layers)
        self.out_channels = cumul_channels + growth_rate

    def forward(self, x):
        """
        Feed the input feature map x through the L subblocks 
        of the DenseBlock.
        """
        out = self.block(x)
        return out

class TransitionLayer(nn.Module):
    """
    This layer is placed between consecutive Dense blocks. 
    It allows the network to downsample the size of feature 
    maps using the pooling operator.

    Concretely, this layer is a composition of 3 operations:
    BN - Conv (1x1) - AveragePool
    
    Additionally, this layer can perform compression by reducing
    the number of output feature maps using a compression factor
    theta.
    """
    def __init__(self, in_channels, theta=0.5, p=0):
        """
        Initialize the different parts of the TransitionBlock.

        Params
        ------
        - in_channels: number of input channels.
        - theta: compression factor in the range [0, 1]. Set to 0.5
          in the paper when using DenseNet-BC.
        """
        super(TransitionLayer, self).__init__()
        self.p = p
        self.out_channels = int(math.floor(theta*in_channels))

        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, 
                              self.out_channels, 
                              kernel_size=1)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        out = self.pool(self.conv(F.relu(self.bn(x))))
        if self.p > 0:
            out = F.dropout(out, p=self.p, training=self.training)
        return out


class DenseNet(nn.Module):
    """
    Densely Connected Convolutional Neural Network [1].

    Connects each layer to every other layer in a feed-forward 
    fashion. This alleviates the vanishing-gradient problem, 
    strengthens feature propagation, encourages feature reuse, and 
    substantially reduces the number of parameters.

    Architecture
    ------------
    * Initial Convolution Layer
    * DenseBlock - TransitionLayer (x2)
    * DenseBlock - Global Avg Pooling
    * Fully Connected
    * Softmax
    
    When we say we have a DenseNet of L layers, L is computed as 
    follows:
    - There are 3 Dense blocks, each with n layers.
    - There is an initial conv layer, and final fully-connected layer.
    - There are 2 Transition layers, each with 1 layer.
    Hence, L = 3*n + 2 + 2 = 3*n + 4.

    This is equivalent to saying (L - 4) must be divisible by 3.

    References
    ----------
    - [1]: Huang et. al., https://arxiv.org/abs/1608.06993
    """
    def __init__(self, 
                 num_blocks, 
                 num_layers_total, 
                 growth_rate, 
                 num_classes, 
                 bottleneck, 
                 p, 
                 theta):
        """
        Initialize the DenseNet network. He. et al weight initialization 
        is used (scaling by sqrt(2/n) to make variance 2/n).

        Params
        ------
        - num_blocks: (int) number of dense blocks in the network. On the CIFAR 
          datasets, this is set to 3 while on ImageNet, it's set to 4.
        - num_layers_total: (int) total number of layers L in the network. L must
          follow the following equation: L = 3*n + 4 where n is the number of
          layers in each dense block.
        - growth_rate: (int) this is k in the paper. Number of feature maps produced
          by each convolution in the dense blocks. 
        - num_classes: (int) number of output classes in the dataset.
        - bottleneck: (bool) specifies if the bottleneck variant of the dense block is
          to be used. 
        - p: (float) dropout rate. Used on non-augmented versions of the datasets.
        - theta: (float) compression factor in the range [0, 1]. In the paper, a value
          of 0.5 is used when bottleneck is used.
        """
        super(DenseNet, self).__init__()

        # ensure L relationship talked above 
        error_msg = "[!] Total number of layers must be 3*n + 4..."
        assert (num_layers_total - 4) % 3 == 0, error_msg

        # compute L, the number of layers in each dense block
        # if bottleneck, we need to adjust L by a factor of 2
        num_layers_dense = int((num_layers_total - 4) / 3)
        if bottleneck:
            num_layers_dense = int(num_layers_dense / 2)

        # ================================== #
        # initial convolutional layer
        out_channels = 16
        if bottleneck:
            out_channels = 2 * growth_rate
        self.conv = nn.Conv2d(3,
                              out_channels, 
                              kernel_size=3,
                              padding=1)
        # ================================== #

        # ================================== #
        # dense blocks and transition layers 
        blocks = []
        for i in range(num_blocks - 1):
            # dense block
            dblock = DenseBlock(num_layers_dense, 
                                out_channels, 
                                growth_rate, 
                                bottleneck, 
                                p)
            blocks.append(dblock)

            # transition block
            out_channels = dblock.out_channels
            trans = TransitionLayer(out_channels, theta, p)
            blocks.append(trans)
            out_channels = trans.out_channels
        # ================================== #

        # ================================== #
        # last dense block does not have transition layer
        dblock = DenseBlock(num_layers_dense, 
                            out_channels, 
                            growth_rate, 
                            bottleneck, 
                            p)
        blocks.append(dblock)
        self.block = nn.Sequential(*blocks)
        self.out_channels = dblock.out_channels
        # ================================== #

        # ================================== #
        # fully-connected layer
        self.fc = nn.Linear(self.out_channels, num_classes)
        # ================================== #

        # ================================== #
        # He et. al weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
        # ================================== #

    def forward(self, x):
        """
        Run the forward pass of the DenseNet model.
        """
        out = self.conv(x)
        out = self.block(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.out_channels)
        out = self.fc(out)
        return out


class DenseStart(nn.Module):
    def __init__(self, out_channels=16, growth_rate=32, bottleneck=False):
        super(DenseStart, self).__init__()

        # ================================== #
        # initial convolutional layer
        self.out_channels = out_channels
        self.growth_rate = growth_rate
        if bottleneck:
            self.out_channels = 2 * self.growth_rate
        self.conv = nn.Conv2d(3,
                              self.out_channels, 
                              kernel_size=3,
                              padding=1)

    def forward(self, x):
        out = self.conv(x)
        return out


# From: https://raw.githubusercontent.com/kevinzakka/densenet/master/layers.py
#########################################################################

class DenseBlockOld(nn.Module):
    def __init__(self, in_size, out_size, kernel, stride):
        super(DenseBlockOld, self).__init__()
        pad_size = kernel // 2
        self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, kernel,
                                             stride=stride, padding=pad_size,
                                             bias=False),
                                   nn.BatchNorm2d(out_size),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(in_size, out_size, kernel,
                                             stride=stride, padding=pad_size,
                                             bias=False),
                                   nn.BatchNorm2d(out_size))
        self.relu = nn.ReLU(inplace=True)
        self.transition = nn.Sequential(nn.BatchNorm2d(in_size),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(in_size, out_size, kernel,
                                                  stride=stride, bias=False),
                                        nn.AdaptiveAvgPool2d(kernel_size=kernel,
                                                             stride=stride))

    def forward(self, inputs1, inputs2):
        x = self.conv1(inputs1)
        in_data = [x, inputs2]
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
            # in_data[small_ch_id] = torch.cat(
                # [in_data[small_ch_id], tmp * 0], 1)
        out = torch.cat([in_data[0], in_data[1]])
        return self.relu(out)

    def _bn_function_factory(self, norm, relu, conv):
        def bn_function(*inputs):
            concated_features = torch.cat(inputs, 1)
            bottleneck_output = conv(relu(norm(concated_features)))
            return bottleneck_output

        return bn_function


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

        elif arch_type == 'densenet':
             for name, in1, in2 in self.cgp:
                if name == 'input' in name:
                    ##########
                    # New: create input layer for DenseNet
                    self.encode.append(DenseStart(out_size))
                    ##########
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
                            self.encode.append(DenseBlock(self.channel_num[in1],
                                                          out_size, kernel))
                                                        #   stride=1))
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
            elif isinstance(layer, DenseBlock):
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
