import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer_helper import BaseDecodeHead

# TODO: rethink this
class DLFSimpleHead(BaseDecodeHead):

    def __init__(self, **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        self.unsample = upsample
        self.out_channels = {'x1': self.channels, 'x2': self.channels * 2,
            'x4': self.channels * 4}[upsample]

        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)

        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.out_channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        self.fusion_conv = ConvModule(
            in_channels=self.out_channels * num_inputs,
            out_channels=self.out_channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

        self.up_conv1 = nn.Sequential(
            nn.ConvTranspose2d(self.out_channels, self.out_channels // 2, 2, stride=2),
            nn.GroupNorm(1, self.out_channels // 2),
            nn.Conv2d(self.out_channels // 2, self.out_channels // 2, 1),
            nn.GroupNorm(1, self.out_channels // 2),
            nn.GELU()
        )

        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose2d(self.out_channels // 2, self.out_channels // 4, 2, stride=2),
            nn.GroupNorm(1, self.out_channels // 4),
            nn.Conv2d(self.out_channels // 4, self.out_channels // 4, 1),
            nn.GroupNorm(1, self.out_channels // 4),
            nn.GELU()
        )

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            outs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        out = self.fusion_conv(torch.cat(outs, dim=1))
        if self.unsample == 'x2':
            out = self.up_conv1(out)

        if self.unsample == 'x4':
            out = self.up_conv2(self.up_conv1(out))

        out = self.cls_seg(out)

        return out
