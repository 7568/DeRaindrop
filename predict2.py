# -*- coding: UTF-8 -*-
"""
Created by root at 2021/3/31
Description:
"""

import torch
import argparse
import os
import numpy as np
import cv2
from models import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--gt_dir", type=str)
    args = parser.parse_args()
    return args



def predict(image):
    image = np.array(image, dtype='float32') / 255.
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :, :, :]
    image = torch.from_numpy(image)
    image = image.to(device)

    out = model_load(image,times_in_attention)[-1]

    out = out.cpu().data
    out = out.numpy()
    out = out.transpose((0, 2, 3, 1))
    out = out[0, :, :, :] * 255.

    return out

if __name__ == '__main__':
    args = get_args()
    args.input_dir = '/home/louis/Documents/git/DeRaindrop/data/test_a/data/'  # 带雨滴的图片的路径
    args.output_dir = '/home/louis/Documents/git/DeRaindrop/data/test_a/result/'  # 图片的路径
    model_weights = '/home/louis/Documents/git/DeRaindrop/models/generator_1617187768.218444.pth.tar'
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    model_load = Generator().to(device)
    checkpoint = torch.load(model_weights)
    model_load.load_state_dict(checkpoint['state_dict'])

    times_in_attention = 4  # attention中提取M的次数

    input_list = sorted(os.listdir(args.input_dir))
    num = len(input_list)
    for i in range(num):
        print('Processing image: %s' % (input_list[i]))
        img = cv2.imread(args.input_dir + input_list[i])

        result = predict(img)
        img_name = input_list[i].split('.')[0]
        cv2.imwrite(args.output_dir + img_name + '.jpg', result)