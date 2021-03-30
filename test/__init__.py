# -*- coding: UTF-8 -*-
"""
Created by louis at 2021/3/25
Description: 反卷积测试
"""
import torch
import  torch.nn as nn
import numpy as np
# input = torch.randn(1, 1, 10, 10)
# input = torch.round(torch.mul(input,10))
# print(input)
# downsample = nn.Conv2d( 1,1,3, stride=1, padding=0)
# w = torch.randn(1, 1, 3, 3)
# w = w/w
# print(w)
# downsample.weight=torch.nn.Parameter(w)
# print(downsample)
# upsample = nn.ConvTranspose2d(1, 1, 3, stride=1, padding=2)
# upsample.weight=torch.nn.Parameter(w)
# h = downsample(input)
# print(torch.round(h))
#
# output = upsample(input)
# print(torch.round(output))

# a=torch.tensor([1,2,3])
# b=torch.tensor([4,5,6])
# print(a * b)

a=range(27)
b=range(10,64,2)
a=np.array(a)
b=np.array(b)
a=np.reshape(a,[3,3,3])
b=np.reshape(b,[3,3,3])
# print(a)
# print('======================')
# print(np.mean(a,1))
#
# print('======================')
# print(b)
# print('======================')
# print(np.mean(b,1))

c = np.abs(np.mean(a,1)-np.mean(b,1))
print(c)
print('======================')
c[c<=31]=0
c[c>31]=1
print(c)

