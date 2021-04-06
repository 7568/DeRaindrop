# -*- coding: UTF-8 -*-
"""
Created by louis at 2021/3/25
Description:
"""
# PyTorch lib
import argparse
import math
import os

import cv2
# Tools lib
import numpy as np
import torch.functional as nn

# Models lib
# Metrics lib
from models import *
import time
from metrics import calc_psnr, calc_ssim

from torch.utils.tensorboard import SummaryWriter


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


# def do_generator(image, times_in_attention):
#     image = np.array(image, dtype='float32') / 255.
#     image = image.transpose((2, 0, 1))
#     image = image[np.newaxis, :, :, :]
#     image = torch.from_numpy(image)
#     image = Variable(image).to(device)
#
#     out_tensor = generator(image, times_in_attention)[-1]
#     out_array = out_tensor
#     out_array = out_array.cpu().data
#     out_array = out_array.numpy()
#     out_array = out_array.transpose((0, 2, 3, 1))
#     out_array = out_array[0, :, :, :] * 255.
#
#     return out_tensor,out_array

def prepare_img_to_tensor(image):
    image = np.array(image, dtype='float32') / 255.
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :, :, :]
    image = torch.from_numpy(image)
    image = image.to(device)
    return image


def do_discriminator(input):
    """
    :param input:
    :return:
    """
    out = discriminator(input)
    return out


def loss_generator(generator_results, back_ground_truth, binary_mask):
    """

    :param generator_results: 生成网络中返回的结果，包括Attention中每一步的M，attention_map,frame1, frame2, 和最终的输出x
    :param back_ground_truth: 干净的背景图片
    :param binary_mask: 原始图片-干净的背景图片，然后取绝对值，再遍历每个元素，大于36的为1，否则为0
    :return:
    """

    mseloss = nn.MSELoss()
    # 计算公式4
    l_att_a_m = 0
    _attention = generator_results[0]
    for i in range(len(_attention)):
        _pow = torch.tensor(math.pow(sida_in_attention, len(_attention) - i - 1)).to(device)
        l_att_a_m += _pow * mseloss(binary_mask, _attention[i])
    l_att_a_m = torch.mean(l_att_a_m)

    # 计算公式5
    _s = [generator_results[1], generator_results[2], generator_results[4]]
    _t = [prepare_img_to_tensor(resize_image(back_ground_truth, 0.25)),
          prepare_img_to_tensor(resize_image(back_ground_truth, 0.5)), prepare_img_to_tensor(back_ground_truth)]
    _lamda = lamda_in_autoencoder
    lm_s_t = 0
    for i in range(len(_s)):
        lm_s_t += _lamda[i] * mseloss(_s[i], _t[i])
    lm_s_t = torch.mean(lm_s_t)

    # 计算公式6
    lp_o_t = 0
    # loss2 = nn.MSELoss()
    vgg_to_gen = vgg16(generator_results[4])
    vgg_to_gt = vgg16(prepare_img_to_tensor(back_ground_truth))
    for i in range(len(vgg_to_gen)):
        lp_o_t += mseloss(vgg_to_gen[i], vgg_to_gt[i])
    lp_o_t = torch.mean(lp_o_t)

    # 计算公式7
    # LGAN(O) = log(1 - D(G(I)))
    l_g = l_att_a_m + lm_s_t + lp_o_t
    if l_g < 0:
        print(l_g)
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


def get_binary_mask(img, back_gt):
    """
    获得公式中的M
    :param img: 带有雨滴的图片
    :param back_gt: 干净的背景图片
    :return:
    """
    _mean_image = np.mean(img, 2)
    _mean_back_gt = np.mean(back_gt, 2)
    _diff = np.abs(_mean_image - _mean_back_gt)
    _diff[_diff <= 28] = 0
    _diff[_diff > 28] = 1
    # torch.from_numpy(_diff zeng)
    _diff = _diff[np.newaxis, np.newaxis, :, :]
    return torch.from_numpy(_diff).to(device)


def loss_adversarial(result, back_gt):
    """
    (8)
    :param result: generator output
    :param d_o: O = G(I) G:generator I:image with raindrop  d_o : D(O)
    :param back_gt:
    :return:
    """
    mseloss = nn.MSELoss()
    d_o = discriminator(result[4])
    d_r = discriminator(prepare_img_to_tensor(back_gt))
    zeros = torch.zeros(d_r[0].size(0), d_r[0].size(1), d_r[0].size(2), d_r[0].size(3)).to(device)
    l_o_r_an = mseloss(d_o[0], result[0][3]) + mseloss(d_r[0], zeros)
    l_o_r_an = torch.mean(l_o_r_an)
    ones = torch.ones(d_o[1].size(0)).to(device)
    loss2 = -torch.log(d_r[1]) - torch.log(ones - d_o[1]) + discriminative_loss_r * l_o_r_an

    return loss2


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def get_analysis(img_path, gt_path):
    img = cv2.imread(img_path)
    # img = cv2.imread(args.input_dir + input_list[_i])
    gt = cv2.imread(gt_path)
    # gt = cv2.imread(args.gt_dir + gt_list[_i])

    img_tensor = prepare_img_to_tensor(img)
    with torch.no_grad():
        out = generator(img_tensor, times_in_attention, device)[-1]
        out = out.cpu().data
        out = out.numpy()
        out = out.transpose((0, 2, 3, 1))
        out = out[0, :, :, :] * 255.
        out = np.array(out, dtype='uint8')
        cur_psnr = calc_psnr(out, gt)
        cur_ssim = calc_ssim(out, gt)
    return cur_psnr, cur_ssim


def write_tensorboard(args):
    input_list = sorted(os.listdir(args.input_dir))
    gt_list = sorted(os.listdir(args.gt_dir))
    test_input_list = sorted(os.listdir(args.test_input_list))
    test_gt_list = sorted(os.listdir(args.test_gt_list))
    train_cumulative_psnr = 0
    test_cumulative_psnr = 0
    train_cumulative_ssim = 0
    test_cumulative_ssim = 0
    # 从训练集和测试集中分别随机抽取10张图片进行结果分析
    sample_train = torch.randint(0, len(input_list), (10,))
    sample_test = torch.randint(0, len(test_input_list), (10,))

    for _i in sample_train:
        # 计算训练集结果
        train_cur_psnr, train_cur_ssim = get_analysis(args.input_dir + input_list[_i], args.gt_dir + gt_list[_i])
        train_cumulative_psnr += train_cur_psnr
        train_cumulative_ssim += train_cur_ssim

    for _i in sample_test:
        # 计算训练集结果
        test_cur_psnr, test_cur_ssim = get_analysis(args.test_input_list + test_input_list[_i],
                                                    args.test_gt_list + test_gt_list[_i])
        test_cumulative_psnr += test_cur_psnr
        test_cumulative_ssim += test_cur_ssim

    writer.add_scalars('PSNR', {'train_PSNR': train_cumulative_psnr / 10, 'test_PSNR': test_cumulative_psnr / 10})
    writer.add_scalars('SSIM', {'train_SSIM': train_cumulative_ssim / 10, 'test_SSIM': test_cumulative_ssim / 10})
    # print('In testing dataset, PSNR is %.4f and SSIM is %.4f' % (cumulative_psnr / _e, cumulative_ssim / _e))


def train():
    input_list = sorted(os.listdir(args.input_dir))
    gt_list = sorted(os.listdir(args.gt_dir))

    optimizer_g = torch.optim.Adam(generator.parameters(), lr=learning_rate)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    for _e in range(epoch):

        for _i in range(len(input_list)):  # 默认一个iteration只有一张图片
            # print('_i = ', _i)
            print('Processing image: %s' % (input_list[_i]))
            _start = time.time()
            generator.zero_grad()
            img = cv2.imread(args.input_dir + input_list[_i])
            gt = cv2.imread(args.gt_dir + gt_list[_i])

            # resize image
            dsize = (360, 240)
            img = cv2.resize(img, dsize)
            gt = cv2.resize(gt, dsize)
            binary_mask = get_binary_mask(img, gt)
            # img = align_to_four(img)
            # gt = align_to_four(gt)
            img_tensor = prepare_img_to_tensor(img)

            optimizer_g.zero_grad()
            result = generator(img_tensor, times_in_attention, device)
            loss1 = loss_generator(result, gt, binary_mask)
            d1 = discriminator(result[4])
            ones = torch.ones(d1[1].size(0)).to(device)
            loss1 += torch.log(ones - d1[1][0])[0]
            # Backpropagation
            loss1.backward()
            optimizer_g.step()

            optimizer_d.zero_grad()
            result2 = generator(img_tensor, times_in_attention, device)

            loss2 = loss_adversarial(result2, gt)
            # Backpropagation
            loss2.backward()
            optimizer_d.step()

        # print(generator.state_dict())
        write_tensorboard(args)
        torch.save({'state_dict': generator.state_dict()},
                   f'/home/louis/Documents/git/DeRaindrop/models/{_e}_generator_{time.time()}.pth.tar')
        torch.save({'state_dict': discriminator.state_dict()},
                   f'/home/louis/Documents/git/DeRaindrop/models/{_e}_discriminator_{time.time()}.pth.tar')

    print("======finish!==========")


if __name__ == '__main__':
    args = get_args()
    args.input_dir = './data/train/rain_image/'  # 带雨滴的图片的路径
    args.gt_dir = './data/train/clean_image/'  # 干净的图片的路径
    args.test_gt_list = './data/train/clean_image/'  # 测试集带雨滴的图片的路径
    args.test_input_list = './data/train/clean_image/'  # 测试集干净的图片的路径
    model_weights = './models/vgg16-397923af.pth'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    vgg16 = Vgg(vgg_init(device, model_weights))

    print(vgg16)
    epoch = 100
    learning_rate = 0.0002
    beta1 = 0.5
    sida_in_attention = 0.8  # attention中的参数sida
    times_in_attention = 4  # attention中提取M的次数
    lamda_in_autoencoder = [0.6, 0.8, 1.0]
    discriminative_loss_r = 0.05
    writer = SummaryWriter()
    train()
