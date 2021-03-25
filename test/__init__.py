# -*- coding: UTF-8 -*-
"""
Created by louis at 2021/3/25
Description: 反卷积测试
"""
import torch
import  torch.nn as nn
input = torch.randn(1, 1, 10, 10)
input = torch.round(torch.mul(input,10))
print(input)
downsample = nn.Conv2d( 1,1,3, stride=1, padding=0)
w = torch.randn(1, 1, 3, 3)
w = w/w
print(w)
downsample.weight=torch.nn.Parameter(w)
print(downsample)
upsample = nn.ConvTranspose2d(1, 1, 3, stride=1, padding=2)
upsample.weight=torch.nn.Parameter(w)
h = downsample(input)
print(torch.round(h))

output = upsample(input)
print(torch.round(output))

