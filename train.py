# -*- coding: UTF-8 -*-
"""
Created by louis at 2021/3/25
Description:
"""
# PyTorch lib
import argparse
import os

import cv2
# Tools lib
import numpy as np
import torch
import torch.functional as nn
from torch.autograd import Variable
import math

# Models lib
# Metrics lib
from metrics import calc_psnr, calc_ssim
from models import *
import sklearn.metrics as sk_metrics


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--gt_dir", type=str)
    args = parser.parse_args()
    return args


def align_to_four(img):
    # print ('before alignment, row = %d, col = %d'%(img.shape[0], img.shape[1]))
    # align to four
    a_row = int(img.shape[0] / 4) * 4
    a_col = int(img.shape[1] / 4) * 4
    img = img[0:a_row, 0:a_col]
    # print ('after alignment, row = %d, col = %d'%(img.shape[0], img.shape[1]))
    return img


def do_generator(image, times_in_attention):
    image = np.array(image, dtype='float32') / 255.
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :, :, :]
    image = torch.from_numpy(image)
    image = Variable(image).to(device)

    out = generator(image, times_in_attention)[-1]

    out = out.cpu().data
    out = out.numpy()
    out = out.transpose((0, 2, 3, 1))
    out = out[0, :, :, :] * 255.

    return out


def do_discriminator(input):
    """
    :param input:
    :return:
    """
    out = discriminator(input)
    return out


def loss_generator(generator_results, back_ground_truth):
    """

    :param generator_results: 生成网络中返回的结果，包括Attention中每一步的M，attention_map,frame1, frame2, 和最终的输出x
    :param back_ground_truth: 干净的背景图片
    :return:
    """

    # 计算公式4
    l_att_a_m = 0
    for i in range(len(generator_results[0])):
        _attention = generator_results[0]
        _mask = _attention[i]
        _a_t = generator_results[1][i]
        l_att_a_m += math.pow(sida_in_attention, len(_attention) - i) * sk_metrics.mean_squared_error(_mask, _a_t)

    # 计算公式6
    generator_output = generator_results[4]
    vgg_to_gt = vgg16(torch.tensor(back_ground_truth))
    vgg_to_gen = vgg16(torch.tensor(generator_output))
    lp_o_t = sk_metrics.mean_squared_error(vgg_to_gt, vgg_to_gen)

    # 计算公式5
    _s = torch.tensor([generator_results[1], generator_results[2], generator_results[3]])
    _t = torch.tensor([resize_image(back_ground_truth, 0.25), resize_image(back_ground_truth, 0.5), back_ground_truth])
    _lamda = lamda_in_autoencoder
    lm_s_t = torch.sum(_lamda * _s * _t)

    # 计算公式7
    # LGAN(O) = log(1 - D(G(I)))
    # l_gan = nn.BCELoss(do_discriminator(generator_results[4]), back_ground_truth)
    l_g = l_att_a_m + lm_s_t + lp_o_t
    return l_g


def resize_image(image, scale_coefficient):
    """
    等比例缩放图片，
    :param image:
    :param scale_coefficient: 缩放系数，例如缩放到一半，则scale_coefficient=0.5
    :return:
    """
    # calculate the 50 percent of original dimensions
    width = int(image.shape[1] * scale_coefficient)
    height = int(image.shape[0] * scale_coefficient)

    # dsize
    dsize = (width, height)

    # resize image
    output = cv2.resize(image, dsize)
    return output


def loss_adversarial(adver_out, back_gt):
    return nn.BCELoss(adver_out, back_gt) * 0.01


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def train():
    input_list = sorted(os.listdir(args.input_dir))
    gt_list = sorted(os.listdir(args.gt_dir))
    num = len(input_list)
    cumulative_psnr = 0
    cumulative_ssim = 0

    optimizer_g = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(beta1, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(beta1, 0.999))
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    criterion = nn.BCELoss()

    for _e in range(epoch):
        for _i in range(num):  # 默认一个iteration只有一张图片
            print('Processing image: %s' % (input_list[_i]))
            generator.zero_grad()
            img = cv2.imread(args.input_dir + input_list[_i])
            gt = cv2.imread(args.gt_dir + gt_list[_i])
            img = align_to_four(img)
            gt = align_to_four(gt)

            # 计算生成网络的损失
            # 生成网络的loss包括
            # 1 . The loss function in each recurrent
            # block is defined as the mean squared error (MSE) between
            # the output attention map at time step t, orAt, and the binary
            # mask, M.
            optimizer_g.zero_grad()
            result = do_generator(img, times_in_attention)
            loss1 = loss_generator(result, gt)
            # Backpropagation
            loss1.backward()
            optimizer_g.step()

            optimizer_d.zero_grad()
            loss2 = loss_adversarial(discriminator(result[4]), gt)
            # Backpropagation
            loss2.backward()
            optimizer_d.step()

            # result = np.array(result, dtype='uint8')
            # cur_psnr = calc_psnr(result, gt)
            # cur_ssim = calc_ssim(result, gt)
            # print('PSNR is %.4f and SSIM is %.4f' % (cur_psnr, cur_ssim))
            # cumulative_psnr += cur_psnr
            # cumulative_ssim += cur_ssim

        # print('In testing dataset, PSNR is %.4f and SSIM is %.4f' % (cumulative_psnr / num, cumulative_ssim / num))


if __name__ == '__main__':
    args = get_args()
    args.input_dir = '/Users/louis/Documents/git/DeRaindrop/demo/input/'  # 带雨滴的图片的路径
    args.gt_dir = '/Users/louis/Documents/git/DeRaindrop/demo/output/'  # 干净的图片的路径
    model_weights = '/Users/louis/Documents/git/DeRaindrop/models/vgg16-397923af.pth'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    vgg16 = Vgg(vgg_init(device, model_weights))

    print(vgg16)
    epoch = 1
    learning_rate = 0.001
    beta1 = 0.5
    sida_in_attention = 0.8  # attention中的参数sida
    times_in_attention = 4  # attention中提取M的次数
    lamda_in_autoencoder = [0.6, 0.8, 1.0]

    train()
