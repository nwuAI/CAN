##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 整个网络的结构
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


import numpy as np
import torch.nn as nn
import Modules.Blockmodule as bm
import Modules.SEmodule as se
import torch.nn.functional as F
import torch
import pdb
import Modules.channel_attention as channel
import Modules.grid_attention as grid
import Modules.scale1 as scale
import Modules.non_local as sa1
import Modules.modules as modules

# 二维切片的参数，2s张和单张的参数
params = {'num_filters': 64,
            'kernel_h': 5,
            'kernel_w': 5,
            'kernel_c': 1,
            'stride_conv': 1,
            'pool': 2,
            'stride_pool': 2,
          # Valid options : NONE, CSE, SSE, CSSE
            'se_block': "CSSE",
            'drop_out': 0.1}


class Backbone(nn.Module):
    def __init__(self, out_channels, num_slices, se_loss=True):
        super(Backbone, self).__init__()

        # 2s张-image encoder.定义其通道数为2s
        params['num_channels'] = int(2 * num_slices)  # batch_size 11 256 256
        self.encode3D1 = bm.EncoderBlock(params, se_block_type='NONE')  # input size 256 256 #11
        params['num_channels'] = 64
        self.encode3D2 = bm.EncoderBlock(params, se_block_type='NONE')  # 128 128 #64
        self.encode3D3 = bm.EncoderBlock(params, se_block_type='NONE')  # 64 64 #64
        self.encode3D4 = bm.EncoderBlock(params, se_block_type='NONE')  # 32 32 #64
        self.bottleneck3D = bm.DenseBlock(params, se_block_type='NONE')  # output size 16 16 #64

        # 2s张-skull encoder,定义其通道数为2s
        params['num_channels'] = int(2 * num_slices)  # batch_size 11 256 256
        self.encode3Ds1 = bm.EncoderBlock(params, se_block_type='NONE')  # input size 256 256 #11
        params['num_channels'] = 64
        self.encode3Ds2 = bm.EncoderBlock(params, se_block_type='NONE')  # 128 128 #64
        self.encode3Ds3 = bm.EncoderBlock(params, se_block_type='NONE')  # 64 64 #64
        self.encode3Ds4 = bm.EncoderBlock(params, se_block_type='NONE')  # 32 32 #64
        self.bottlenecks3D = bm.DenseBlock(params, se_block_type='NONE')  # output size 16 16 #64

        # 单张二维切片 定义其通道数为1
        params['num_channels'] = 1  # batch_size 11 256 256
        self.encode2D1 = bm.EncoderBlock(params, se_block_type='NONE')  # input size 256 256 #11
        params['num_channels'] = 64
        self.encode2D2 = bm.EncoderBlock(params, se_block_type='NONE')  # 128 128 #64
        self.encode2D3 = bm.EncoderBlock(params, se_block_type='NONE')  # 64 64 #64
        self.encode2D4 = bm.EncoderBlock(params, se_block_type='NONE')  # 32 32 #64
        self.bottleneck2D = bm.DenseBlock(params, se_block_type='NONE')  # output size 16 16 #64

        # 三个双门控空间注意力
        self.attentionblock1 = grid.MultiAttentionBlock(
            in_size=64, gate_size=64, inter_size=64,
            nonlocal_mode='concatenation',
            sub_sample_factor=(1, 1))
        self.attentionblock2 = grid.MultiAttentionBlock(
            in_size=64, gate_size=64, inter_size=64,
            nonlocal_mode='concatenation',
            sub_sample_factor=(1, 1))
        self.attentionblock3 = grid.MultiAttentionBlock(
            in_size=64, gate_size=64, inter_size=64,
            nonlocal_mode='concatenation',
            sub_sample_factor=(1, 1))

        # 单个non-local注意力
        self.sa1 = sa1.NONLocalBlock2D(in_channels=64, inter_channels=64 // 4)

        # 四个编码与解码的拼接，注意这是解码的低层维度上采样之后与编码的特征拼接
        self.up_contact4 = modules.UpCat(64, 64, is_deconv=True)
        self.up_contact3 = modules.UpCat(64, 64, is_deconv=True)
        self.up_contact2 = modules.UpCat(64, 64, is_deconv=True)
        self.up_contact1 = modules.UpCat(64, 64, is_deconv=True)

        # 四个通道注意力
        self.CA4 = channel.SE_Conv_Block(64, 64, drop_out=True)
        self.CA3 = channel.SE_Conv_Block(64, 64)
        self.CA2 = channel.SE_Conv_Block(64, 64)
        self.CA1 = channel.SE_Conv_Block(64, 64)

        # 四个尺度注意力之前的四个不同尺度特征上采样
        self.dsv4 = modules.UnetDsv3(64, 16, scale_factor=(256, 256))
        self.dsv3 = modules.UnetDsv3(64, 16, scale_factor=(256, 256))
        self.dsv2 = modules.UnetDsv3(64, 16, scale_factor=(256, 256))
        self.dsv1 = modules.UnetDsv3(64, 16, scale_factor=(256, 256))

        # 一个尺度注意力
        self.scale_att = scale.scale_atten_convblock(64, 4)

        # 用到的所有卷积
        self.conv1 = nn.Conv2d(in_channels=128,out_channels=64,kernel_size=(5,5),padding=(2,2),stride=1)
        self.conv2 = nn.Conv2d(in_channels=128,out_channels=64,kernel_size=(5,5),padding=(2,2),stride=1)
        self.conv3 = nn.Conv2d(in_channels=128,out_channels=64,kernel_size=(5,5),padding=(2,2),stride=1)
        self.conv4 = nn.Conv2d(in_channels=128,out_channels=64,kernel_size=(5,5),padding=(2,2),stride=1)
        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False),
                                   nn.Conv2d(64, out_channels, 1))  # label output

    def forward(self, input3D, skull3D,input2D):
        """
        :param input: X
       :return: probabiliy map
        """

        # for 2s-image
        e13D, out1, ind1 = self.encode3D1.forward(input3D)  # 256 256 64
        e23D, out2, ind2 = self.encode3D2.forward(e13D)  # 128 128 64
        e33D, out3, ind3 = self.encode3D3.forward(e23D)  # 64 64 64
        e43D, out4, ind4 = self.encode3D4.forward(e33D)  # 32 32 64
        bn3D = self.bottleneck3D.forward(e43D)  # 32 32 64

        # for 2s-skull
        e1s3D, outs1, inds1 = self.encode3Ds1.forward(skull3D)  # 256 256 64
        e2s3D, outs2, inds2 = self.encode3Ds2.forward(e1s3D)  # 128 128 64
        e3s3D, outs3, inds3 = self.encode3Ds3.forward(e2s3D)  # 64 64 64
        e4s3D, outs4, inds4 = self.encode3Ds4.forward(e3s3D)  # 32 32 64
        bns3D = self.bottlenecks3D.forward(e4s3D)  # 32 32 64

        # for 1-image
        e12D, out2D1, ind2D1 = self.encode2D1.forward(input2D)  # 256 256 64
        e22D, out2D2, ind2D2 = self.encode2D2.forward(e12D)  # 128 128 64
        e32D, out2D3, ind2D3 = self.encode2D3.forward(e22D)  # 64 64 64
        e42D, out2D4, ind2D4 = self.encode2D4.forward(e32D)  # 32 32 64
        bn2D = self.bottleneck2D.forward(e42D)  # 32 32 64

        # 特征融合，即将每层编码后的2s-image、2s-skull、1-image特征融合
        # 融合方式，（2s-image）*（2s-skull），之后再与1-image进行最大值融合
        fusion1 = torch.max(out1,out2D1)
        fusion12=fusion1*outs1
        fusion2 = torch.max(out2,out2D2)
        fusion22 = fusion2*outs2
        fusion3 = torch.max(out3,out2D3)
        fusion32 = fusion3*outs3
        fusion4 = torch.max(out4,out2D4)
        fusion42 = fusion4*outs4
        fusion5 = torch.max(bn3D,bn2D)
        fusion52 = fusion5*bns3D

        # image decoder
        up4 = self.up_contact4(fusion42, fusion52)
        up4 = self.conv4(up4)
        g_conv4 = self.sa1(up4)

        up4, attw4 = self.CA4(g_conv4)
        g_conv3, att3 = self.attentionblock3(fusion32, up4)
        up3 = self.up_contact3(g_conv3, up4)

        up3 = self.conv3(up3)
        up3, attw3 = self.CA3(up3)
        g_conv2, att2 = self.attentionblock2(fusion22, up3)
        up2 = self.up_contact2(g_conv2, up3)

        up2 = self.conv2(up2)
        up2, attw2 = self.CA2(up2)
        g_conv1, att1 = self.attentionblock1(fusion12, up2)
        up1 = self.up_contact1(g_conv1, up2)

        up1 = self.conv1(up1)
        up1, attw1 = self.CA1(up1)

        # 将不同尺度特征进行上采样，上采样的尺度特征与输入相同
        dsv4 = self.dsv4(up4)
        dsv3 = self.dsv3(up3)
        dsv2 = self.dsv2(up2)
        dsv1 = self.dsv1(up1)

        # 拼接四个上采样后的特征
        dsv_cat = torch.cat([dsv1, dsv2, dsv3, dsv4], dim=1)
        # 进行尺度注意力运算
        outimage = self.scale_att(dsv_cat)
        # 最后经过卷积获取输出结果

        out_label = self.conv6(outimage)  # n 28 256 256
        # 返回最终结果
        return out_label