import argparse
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision as tv
from impaint import impaint
import DeepFillv2.network as network

# ----------------------------------------
#        Initialize the parameters
# ----------------------------------------
def init():

    parser = argparse.ArgumentParser()
    # General parameters
    parser.add_argument('--results_path', type=str, default='./results', help='testing samples path that is a folder')
    parser.add_argument('--gan_type', type=str, default='WGAN', help='the type of GAN for training')
    parser.add_argument('--gpu_ids', type=str, default="1", help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--cudnn_benchmark', type=bool, default=True, help='True for unchanged input data type')
    # Training parameters
    parser.add_argument('--epoch', type=int, default=40, help='number of epochs of training')
    parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='number of cpu threads to use during batch generation')
    # Network parameters
    parser.add_argument('--in_channels', type=int, default=4, help='input RGB image + 1 channel mask')
    parser.add_argument('--out_channels', type=int, default=3, help='output RGB image')
    parser.add_argument('--latent_channels', type=int, default=48, help='latent channels')
    parser.add_argument('--pad_type', type=str, default='zero', help='the padding type')
    parser.add_argument('--activation', type=str, default='elu', help='the activation type')
    parser.add_argument('--norm', type=str, default='none', help='normalization type')
    parser.add_argument('--init_type', type=str, default='normal', help='the initialization type')
    parser.add_argument('--init_gain', type=float, default=0.02, help='the initialization gain')

    opt = parser.parse_args()

    return opt


# ----------------------------------------
#                 Network
# ----------------------------------------
def create_generator(opt):
    # Initialize the networks
    generator = network.GatedGenerator(opt)
    print('Generator is created!')
    network.weights_init(generator, init_type = opt.init_type, init_gain = opt.init_gain)
    print('Initialize generator with %s type' % opt.init_type)
    return generator


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_img(opt, img_copy):
    # Save to certain path
    save_img_name = './background/background.png'

    cv2.imwrite(save_img_name, img_copy)

## for contextual attention

def extract_image_patches(images, ksizes, strides, rates, padding='same'):
    """
    Extract patches from images and put them in the C output dimension.
    :param padding:
    :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
    :param ksizes: [ksize_rows, ksize_cols]. The size of the sliding window for
     each dimension of images
    :param strides: [stride_rows, stride_cols]
    :param rates: [dilation_rows, dilation_cols]
    :return: A Tensor
    """
    assert len(images.size()) == 4
    assert padding in ['same', 'valid']
    batch_size, channel, height, width = images.size()

    if padding == 'same':
        images = same_padding(images, ksizes, strides, rates)
    elif padding == 'valid':
        pass
    else:
        raise NotImplementedError('Unsupported padding type: {}.\
                Only "same" or "valid" are supported.'.format(padding))

    unfold = torch.nn.Unfold(kernel_size=ksizes,
                             dilation=rates,
                             padding=0,
                             stride=strides)
    patches = unfold(images)
    return patches  # [N, C*k*k, L], L is the total number of such blocks

def same_padding(images, ksizes, strides, rates):
    assert len(images.size()) == 4
    batch_size, channel, rows, cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows-1)*strides[0]+effective_k_row-rows)
    padding_cols = max(0, (out_cols-1)*strides[1]+effective_k_col-cols)
    # Pad the input
    padding_top = int(padding_rows / 2.)
    padding_left = int(padding_cols / 2.)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    paddings = (padding_left, padding_right, padding_top, padding_bottom)
    images = torch.nn.ZeroPad2d(paddings)(images)
    return images

def reduce_mean(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.mean(x, dim=i, keepdim=keepdim)
    return x


def reduce_std(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.std(x, dim=i, keepdim=keepdim)
    return x


def reduce_sum(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.sum(x, dim=i, keepdim=keepdim)
    return x

#----------------------------------
#        Input processing
#----------------------------------

def concat_masks(masks):
    tot_mask = masks[0]

    if len(masks) == 1:
        return tot_mask.astype(np.float32) * 255

    for m in masks[1:]:
        tot_mask = tot_mask | m

    return tot_mask.astype(np.uint8) * 255

def impainting(image, masks, i):
    full_mask = concat_masks(masks)
    cv2.imwrite('masks/' + str(i) + '.png', masks[2].astype(np.uint8) * 255)
    bg = impaint(image, masks[2].astype(np.uint8) * 255)

    return bg
