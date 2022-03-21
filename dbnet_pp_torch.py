
import logging
from collections import OrderedDict
import os
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import logging
import os
import torch
from torch import nn
from torchsummary import summary
import tensorwatch as tw

class HSwish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out

class ResizeFixedSize:
    def __init__(self, short_size, resize_text_polys=True):
        """
        :param size: resize尺寸,数字或者list的形式，如果为list形式，就是[w,h]
        :return:
        """
        self.short_size = short_size
        self.resize_text_polys = resize_text_polys

    def __call__(self, data: dict) -> dict:
        """
        对图片和文本框进行缩放
        :param data: {'img':,'text_polys':,'texts':,'ignore_tags':}
        :return:
        """
        im = data['img']
        text_polys = data['text_polys']
        h, w, _ = im.shape
        # 如果两个边长度都小于640，那么将长边缩放到640，短边缩放到一个32的倍数
        if max(h, w) < self.short_size:
            if h > w:
                ratio = float(self.short_size) / h
            else:
                ratio = float(self.short_size) / w
        else:
            ratio = 1.
        
        # if min(h, w) < self.short_size:
        #     if h < w:
        #         ratio = float(self.short_size) / h
        #     else:
        #         ratio = float(self.short_size) / w
        # else:
        #     ratio = 1.
        resize_h = int(h * ratio)
        resize_w = int(w * ratio)
        resize_h = max(int(round(resize_h / 32) * 32), 32)
        resize_w = max(int(round(resize_w / 32) * 32), 32)

        try:
            if int(resize_w) <= 0 or int(resize_h) <= 0:
                return None, (None, None)
            img = cv2.resize(im, (int(resize_w), int(resize_h)))
        except:
            print(img.shape, resize_w, resize_h)
            import sys
            sys.exit(0)

        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)
        if self.resize_text_polys:
            text_polys[:, 0] *= ratio_h
            text_polys[:, 1] *= ratio_w

        data['img'] = img
        data['text_polys'] = text_polys
        print(img.shape)
        return data


class ConvBNACT(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, act=None):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, groups=groups,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        if act == 'relu':
            self.act = nn.ReLU()
        elif act == 'hard_swish':
            self.act = HSwish()
        elif act is None:
            self.act = None


    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class SEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hsigmoid_type='others', ratio=4):
        super().__init__()
        num_mid_filter = out_channels // ratio
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=num_mid_filter, kernel_size=1, bias=True)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=num_mid_filter, kernel_size=1, out_channels=out_channels, bias=True)
        self.relu2 = nn.Hardsigmoid(hsigmoid_type)


    def forward(self, x):
        attn = self.pool(x)
        attn = self.conv1(attn)
        attn = self.relu1(attn)
        attn = self.conv2(attn)
        attn = self.relu2(attn)
        return x * attn


class ResidualUnit(nn.Module):
    def __init__(self, num_in_filter, num_mid_filter, num_out_filter, stride, kernel_size, act=None, use_se=False):
        super().__init__()
        self.conv0 = ConvBNACT(in_channels=num_in_filter, out_channels=num_mid_filter, kernel_size=1, stride=1,
                               padding=0, act=act)

        self.conv1 = ConvBNACT(in_channels=num_mid_filter, out_channels=num_mid_filter, kernel_size=kernel_size,
                               stride=stride,
                               padding=int((kernel_size - 1) // 2), act=act, groups=num_mid_filter)
        if use_se:
            self.se = SEBlock(in_channels=num_mid_filter, out_channels=num_mid_filter)
        else:
            self.se = None

        self.conv2 = ConvBNACT(in_channels=num_mid_filter, out_channels=num_out_filter, kernel_size=1, stride=1,
                               padding=0)
        self.not_add = num_in_filter != num_out_filter or stride != 1


    def forward(self, x):
        y = self.conv0(x)
        y = self.conv1(y)
        if self.se is not None:
            y = self.se(y)
        y = self.conv2(y)
        if not self.not_add:
            y = x + y
        return y


class MobileNetV3(nn.Module):
    def __init__(self, in_channels, pretrained=True, **kwargs):
        """
        the MobilenetV3 backbone network for detection module.
        Args:
            params(dict): the super parameters for build network
        """
        super().__init__()
        self.scale = kwargs.get('scale', 0.5)
        model_name = kwargs.get('model_name', 'small')
        self.disable_se=kwargs.get('disable_se','True')
        self.inplanes = 16
        if model_name == "large":
            self.cfg = [
                # k, exp, c,  se,     nl,  s,
                [3, 16, 16, False, 'relu', 1],
                [3, 64, 24, False, 'relu', 2],
                [3, 72, 24, False, 'relu', 1],
                [5, 72, 40, True, 'relu', 2],
                [5, 120, 40, True, 'relu', 1],
                [5, 120, 40, True, 'relu', 1],
                [3, 240, 80, False, 'hard_swish', 2],
                [3, 200, 80, False, 'hard_swish', 1],
                [3, 184, 80, False, 'hard_swish', 1],
                [3, 184, 80, False, 'hard_swish', 1],
                [3, 480, 112, True, 'hard_swish', 1],
                [3, 672, 112, True, 'hard_swish', 1],
                [5, 672, 160, True, 'hard_swish', 2],
                [5, 960, 160, True, 'hard_swish', 1],
                [5, 960, 160, True, 'hard_swish', 1],
            ]
            self.cls_ch_squeeze = 960
            self.cls_ch_expand = 1280
        elif model_name == "small":
            self.cfg = [
                # k, exp, c,  se,     nl,  s,
                [3, 16, 16, True, 'relu', 2],
                [3, 72, 24, False, 'relu', 2],
                [3, 88, 24, False, 'relu', 1],
                [5, 96, 40, True, 'hard_swish', 2],
                [5, 240, 40, True, 'hard_swish', 1],
                [5, 240, 40, True, 'hard_swish', 1],
                [5, 120, 48, True, 'hard_swish', 1],
                [5, 144, 48, True, 'hard_swish', 1],
                [5, 288, 96, True, 'hard_swish', 2],
                [5, 576, 96, True, 'hard_swish', 1],
                [5, 576, 96, True, 'hard_swish', 1],
            ]
            self.cls_ch_squeeze = 576
            self.cls_ch_expand = 1280
        else:
            raise NotImplementedError("mode[" + model_name +
                                      "_model] is not implemented!")

        supported_scale = [0.35, 0.5, 0.75, 1.0, 1.25]
        assert self.scale in supported_scale, \
            "supported scale are {} but input scale is {}".format(supported_scale, self.scale)

        scale = self.scale
        inplanes = self.inplanes
        cfg = self.cfg
        cls_ch_squeeze = self.cls_ch_squeeze
        # conv1
        self.conv1 = ConvBNACT(in_channels=in_channels,
                               out_channels=self.make_divisible(inplanes * scale),
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               groups=1,
                               act='hard_swish')
        i = 0
        inplanes = self.make_divisible(inplanes * scale)
        self.stages = nn.ModuleList()
        block_list = []
        self.out_channels = []
        for layer_cfg in cfg:
            se = layer_cfg[3] and not self.disable_se
            if layer_cfg[5] == 2 and i > 0:
                self.out_channels.append(inplanes)
                self.stages.append(nn.Sequential(*block_list))
                block_list = []
            block = ResidualUnit(num_in_filter=inplanes,
                                 num_mid_filter=self.make_divisible(scale * layer_cfg[1]),
                                 num_out_filter=self.make_divisible(scale * layer_cfg[2]),
                                 act=layer_cfg[4],
                                 stride=layer_cfg[5],
                                 kernel_size=layer_cfg[0],
                                 use_se=se)
            block_list.append(block)
            inplanes = self.make_divisible(scale * layer_cfg[2])
            i += 1
        block_list.append(ConvBNACT(
            in_channels=inplanes,
            out_channels=self.make_divisible(scale * cls_ch_squeeze),
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            act='hard_swish'))
        self.stages.append(nn.Sequential(*block_list))
        self.out_channels.append(self.make_divisible(scale * cls_ch_squeeze))

        if pretrained:
            ckpt_path = f'./weights/MobileNetV3_{model_name}_x{str(scale).replace(".", "_")}.pth'
            logger = logging.getLogger('torchocr')
            if os.path.exists(ckpt_path):
                logger.info('load imagenet weights')
                self.load_state_dict(torch.load(ckpt_path))
            else:
                logger.info(f'{ckpt_path} not exists')

    def make_divisible(self, v, divisor=8, min_value=None):
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v


    def forward(self, x):
        x = self.conv1(x)
        out = []
        for stage in self.stages:
            x = stage(x)
            out.append(x)

        return out


class spatial_attition(nn.Module):
    def __init__(self, in_channels, out_channels=96):
        """
        :param in_channels: 基础网络输出的维度
        :param kwargs:
        """
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(in_channels, 4, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        x1 = self.conv1(avgout)
        x1 = self.relu(x1)
        
        x1 = self.conv2(x1)
        x1 = torch.sigmoid(x1)
        
        x2 = x1 + x
        x2 = self.conv3(x2)
        x2 = torch.sigmoid(x2)
        x2 = x2.unsqueeze(2)
        
        return x2
    	
    	
    	
    	


class ASF(nn.Module):
    def __init__(self, in_channels, out_channels=96, **kwargs):
        """
        :param in_channels: 基础网络输出的维度
        :param kwargs:
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=3, padding=1, bias=False)
        self.sa = spatial_attition(in_channels // 4)

    def forward(self, x):
        x0 = torch.stack(x, 1)

        x = torch.cat(x, dim=1)

        x1 = self.conv1(x)
        x1 = self.sa(x1)
        x1 = x1 + x0
        x1 = x1.view(-1, self.out_channels, x1.shape[-2], x1.shape[-1])
        x1 = torch.sigmoid(x1)
        return x1
    	


class DB_fpn(nn.Module):
    def __init__(self, in_channels, out_channels=96, **kwargs):
        """
        :param in_channels: 基础网络输出的维度
        :param kwargs:
        输出：[b, 96, h//4, w//4]
        """
        super().__init__()
        inplace = True
        self.out_channels = out_channels
        # reduce layers
        self.in2_conv = nn.Conv2d(in_channels[0], self.out_channels, kernel_size=1, bias=False)
        self.in3_conv = nn.Conv2d(in_channels[1], self.out_channels, kernel_size=1, bias=False)
        self.in4_conv = nn.Conv2d(in_channels[2], self.out_channels, kernel_size=1, bias=False)
        self.in5_conv = nn.Conv2d(in_channels[3], self.out_channels, kernel_size=1, bias=False)
        # Smooth layers
        self.p5_conv = nn.Conv2d(self.out_channels, self.out_channels // 4, kernel_size=3, padding=1, bias=False)
        self.p4_conv = nn.Conv2d(self.out_channels, self.out_channels // 4, kernel_size=3, padding=1, bias=False)
        self.p3_conv = nn.Conv2d(self.out_channels, self.out_channels // 4, kernel_size=3, padding=1, bias=False)
        self.p2_conv = nn.Conv2d(self.out_channels, self.out_channels // 4, kernel_size=3, padding=1, bias=False)

    def _upsample_add(self, x, y):
        return F.interpolate(x, scale_factor=2) + y

    def _upsample_cat(self, p2, p3, p4, p5):
        p3 = F.interpolate(p3, scale_factor=2)
        p4 = F.interpolate(p4, scale_factor=4)
        p5 = F.interpolate(p5, scale_factor=8)
        return p5, p4, p3, p2
        # return torch.cat([p5, p4, p3, p2], dim=1)

    def forward(self, x):
        c2, c3, c4, c5 = x
        
        in5 = self.in5_conv(c5)
        in4 = self.in4_conv(c4)
        in3 = self.in3_conv(c3)
        in2 = self.in2_conv(c2)

        out4 = self._upsample_add(in5, in4)
        out3 = self._upsample_add(out4, in3)
        out2 = self._upsample_add(out3, in2)

        p5 = self.p5_conv(in5)
        p4 = self.p4_conv(out4)
        p3 = self.p3_conv(out3)
        p2 = self.p2_conv(out2)

        x = self._upsample_cat(p2, p3, p4, p5)
        return x



class Head(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 4, kernel_size=3, padding=1,
                               bias=False)
        self.conv_bn1 = nn.BatchNorm2d(in_channels // 4, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.ConvTranspose2d(in_channels=in_channels // 4, out_channels=in_channels // 4, kernel_size=2,
                                        stride=2)
        self.conv_bn2 = nn.BatchNorm2d(in_channels // 4, track_running_stats=True)
        self.conv3 = nn.ConvTranspose2d(in_channels=in_channels // 4, out_channels=1, kernel_size=2, stride=2)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv_bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.conv_bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = torch.sigmoid(x)
        return x


class DBHead(nn.Module):
    """
    Differentiable Binarization (DB) for text detection:
        see https://arxiv.org/abs/1911.08947
    args:
        params(dict): super parameters for build DB network
    """

    def __init__(self, in_channels, k=50):
        super().__init__()
        self.k = k
        self.binarize = Head(in_channels)
        self.thresh = Head(in_channels)
        self.binarize.apply(self.weights_init)
        self.thresh.apply(self.weights_init)
        self.training = False

    def step_function(self, x, y):
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))


    def forward(self, x):
        shrink_maps = self.binarize(x)
        """
        if not self.training:
            return shrink_maps"""
        threshold_maps = self.thresh(x)
        binary_maps = self.step_function(shrink_maps, threshold_maps)
        y = torch.cat((shrink_maps, threshold_maps, binary_maps), dim=1)
        print(y.shape)
        return y

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)


class DetModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = MobileNetV3(3)

        self.neck = DB_fpn(self.backbone.out_channels)# neck_dict[neck_type](self.backbone.out_channels, **config.neck)

        self.asf = ASF(self.neck.out_channels)

        self.head = DBHead(self.neck.out_channels) # head_dict[head_type](self.neck.out_channels, **config.head)

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.asf(x)
        x = self.head(x)
        return x

model = DetModel()
a = torch.rand(2, 3, 640, 640)
model(a)
