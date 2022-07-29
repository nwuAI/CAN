##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# DenseNet的代码，在backbone.py中会引入这个
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
import Modules.SEmodule as se
from torch.autograd import Function


# 密集连接块
class DenseBlock(nn.Module):
    """Block with dense connections
    :param params
    :type params: dict
    :param se_block_type: Squeeze and Excite block type to be included, defaults to None
    :type se_block_type: str, valid options are {'NONE', 'CSE', 'SSE', 'CSSE'}, optional
    :return: forward passed tensor
    :rtype: torch.tonsor [FloatTensor]
    """

    def __init__(self, params, se_block_type=None):
        super(DenseBlock, self).__init__()
        # 选择CSE/SSE/CSSE，我们这里没用到这些注意力，所以直接将se_block_type=None即可，后续如果跑这些代码，那就改为对应的名字即可
        if se_block_type == se.SELayer.CSE.value:
            self.SELayer = se.ChannelSELayer(params['num_filters'])

        elif se_block_type == se.SELayer.SSE.value:
            self.SELayer = se.SpatialSELayer(params['num_filters'])

        elif se_block_type == se.SELayer.CSSE.value:
            self.SELayer = se.ChannelSpatialSELayer(params['num_filters'])
        else:
            self.SELayer = None

        padding_h = int((params['kernel_h'] - 1) / 2)
        padding_w = int((params['kernel_w'] - 1) / 2)

        conv1_out_size = int(params['num_channels'] + params['num_filters'])
        conv2_out_size = int(params['num_channels'] + params['num_filters'] + params['num_filters'])

        self.conv1 = nn.Conv2d(in_channels=params['num_channels'], out_channels=params['num_filters'],
                               kernel_size=(
                                   params['kernel_h'], params['kernel_w']),
                               padding=(padding_h, padding_w),
                               stride=params['stride_conv'])  # 步长为1
        self.conv2 = nn.Conv2d(in_channels=conv1_out_size, out_channels=params['num_filters'],
                               kernel_size=(
                                   params['kernel_h'], params['kernel_w']),
                               padding=(padding_h, padding_w),
                               stride=params['stride_conv'])
        self.conv3 = nn.Conv2d(in_channels=conv2_out_size, out_channels=params['num_filters'],
                               kernel_size=(1, 1),
                               padding=(0, 0),
                               stride=params['stride_conv'])
        self.batchnorm1 = nn.BatchNorm2d(num_features=params['num_channels'])
        self.batchnorm2 = nn.BatchNorm2d(num_features=conv1_out_size)
        self.batchnorm3 = nn.BatchNorm2d(num_features=conv2_out_size)
        self.prelu = nn.PReLU()
        if params['drop_out'] > 0:
            self.drop_out_needed = True
            self.drop_out = nn.Dropout2d(params['drop_out'])
        else:
            self.drop_out_needed = False

    def forward(self, input):
        """Forward pass
        :param input: Input tensor, shape = (N x C x H x W)
        :type input: torch.tensor [FloatTensor]
        :return: Forward passed tensor
        :rtype: torch.tensor [FloatTensor]
        """
        # denseblock操作：BN+RELU+conv,三次
        o1 = self.batchnorm1(input)  # 2 11 256 256
        o2 = self.prelu(o1)  # 2 11 256 256
        o3 = self.conv1(o2)  # 2 64 256 256
        o4 = torch.cat((input, o3), dim=1)  # 2 75 256 256#将这两个张量拼接在一起，通道的合并
        o5 = self.batchnorm2(o4)  # 2 75 256 256
        o6 = self.prelu(o5)  # 2 75 256 256
        o7 = self.conv2(o6)  # 2 64 256 256
        o8 = torch.cat((input, o3, o7), dim=1)  # 2 139 256 256
        o9 = self.batchnorm3(o8)  # 2 139 256 256
        o10 = self.prelu(o9)  # 2 139 256 256
        out = self.conv3(o10)  # 2 64 256 256
        return out


# 编码部分
class EncoderBlock(DenseBlock):
    """Dense encoder block with maxpool and an optional SE block，gaibianchicundaxiao
    :param params
    :type params: dict
    :param se_block_type: Squeeze and Excite block type to be included, defaults to None
    :type se_block_type: str, valid options are {'NONE', 'CSE', 'SSE', 'CSSE'}, optional
    :return: output tensor with maxpool, output tensor without maxpool, indices for unpooling
    :rtype: torch.tensor [FloatTensor], torch.tensor [FloatTensor], torch.tensor [LongTensor]
    """

    def __init__(self, params, se_block_type=None):
        super(EncoderBlock, self).__init__(params, se_block_type=se_block_type)
        self.maxpool = nn.MaxPool2d(
            kernel_size=params['pool'], stride=params['stride_pool'], return_indices=True)

    def forward(self, input):
        """Forward pass

        :param input: Input tensor, shape = (N x C x H x W)
        :type input: torch.tensor [FloatTensor]
        :return: output tensor with maxpool, output tensor without maxpool, indices for unpooling
        :rtype: torch.tensor [FloatTensor], torch.tensor [FloatTensor], torch.tensor [LongTensor]
        """

        out_block = super(EncoderBlock, self).forward(input)  # 每次denseblock的输出
        # 选择压缩和激励模块
        if self.SELayer:
            out_block = self.SELayer(out_block)
        # 是否使用dropout
        if self.drop_out_needed:
            out_block = self.drop_out(out_block)
        # 输出编码后的特征与索引数indices,方便后续拼接
        out_encoder, indices = self.maxpool(out_block)  # 下采样的结果
        return out_encoder, out_block, indices


# 解码部分
class DecoderBlock(DenseBlock):
    """Dense decoder block with maxunpool and an optional skip connections and SE block
    :param params
    :type params: dict
    :param se_block_type: Squeeze and Excite block type to be included, defaults to None
    :type se_block_type: str, valid options are {'NONE', 'CSE', 'SSE', 'CSSE'}, optional
    :return: forward passed tensor
    :rtype: torch.tensor [FloatTensor]
    """

    def __init__(self, params, se_block_type=None):
        super(DecoderBlock, self).__init__(params, se_block_type=se_block_type)  # 每层denseblock
        self.unpool = nn.MaxUnpool2d(
            kernel_size=params['pool'], stride=params['stride_pool'])  # 上采样

    def forward(self, input, out_block=None, indices=None):
        """Forward pass
        :param input: Input tensor, shape = (N x C x H x W)
        :type input: torch.tensor [FloatTensor]
        :param out_block: Tensor for skip connection, shape = (N x C x H x W), defaults to None
        :type out_block: torch.tensor [FloatTensor], optional
        :param indices: Indices used for unpooling operation, defaults to None
        :type indices: torch.tensor, optional
        :return: Forward passed tensor
        :rtype: torch.tensor [FloatTensor]
        """

        unpool = self.unpool(input, indices)  # 上采样的结果
        concat = torch.cat((out_block, unpool), dim=1)  # 上采样后的特征与编码部分同级特征拼接
        out_block = super(DecoderBlock, self).forward(concat)

        if self.SELayer:
            out_block = self.SELayer(out_block)

        if self.drop_out_needed:
            out_block = self.drop_out(out_block)
        return out_block

