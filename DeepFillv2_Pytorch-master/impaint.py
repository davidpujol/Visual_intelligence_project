import argparse
import os

from PIL import Image
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import os
import time
import datetime
import numpy as np
import cv2

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from data import Data

import network
import test_dataset
import utils

def impaint(opt):

    # Save the model if pre_train == True
    def load_model_generator(net, epoch, opt):
        model_name = 'deepfillv2_WGAN_G_epoch%d_batchsize%d.pth' % (epoch, 4)
        model_name = os.path.join('pretrained_model', model_name)
        pretrained_dict = torch.load(model_name, map_location=torch.device('cpu'))
        generator.load_state_dict(pretrained_dict)

    # ----------------------------------------
    #      Initialize training parameters
    # ----------------------------------------

    # configurations
    if not os.path.exists(opt.results_path):
        os.makedirs(opt.results_path)

    # Build networks
    generator = utils.create_generator(opt).eval()
    print('-------------------------Loading Pretrained Model-------------------------')
    load_model_generator(generator, opt.epoch, opt)
    print('-------------------------Pretrained Model Loaded-------------------------')

    # To device
    generator = generator.cuda()

    # ----------------------------------------
    #       Initialize training dataset
    # ----------------------------------------

    dataset = Data(opt)
    
    # Define the dataloader
    dataloader = DataLoader(dataset, batch_size = opt.batch_size, shuffle = False, num_workers = opt.num_workers, pin_memory = True)
    
    for batch_id, (data_img, data_mask) in enumerate(dataloader):
        img = data_img
        print(batch_id)
        print(img.size(), flush=True)
        mask = data_mask

        img = img.cuda()
        mask = mask.cuda()

        print(len(img), flush=True)
        print(len(mask), flush=True)

        # Generator output
        with torch.no_grad():
            first_out, second_out = generator(img, mask)

        # forward propagation
        first_out_wholeimg = img * (1 - mask) + first_out * mask  # in range [0, 1]
        second_out_wholeimg = img * (1 - mask) + second_out * mask  # in range [0, 1]

        masked_img = img * (1 - mask) + mask
        mask = torch.cat((mask, mask, mask), 1)

        res = second_out_wholeimg * 255
        # Process img_copy and do not destroy the data of img
        img_copy = res.clone().data.permute(0, 2, 3, 1)[0, :, :, :].cpu().numpy()
        img_copy = np.clip(img_copy, 0, 255)
        img_copy = img_copy.astype(np.uint8)
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR)
        
        # Save to certain path
        save_img_name = 'result_' + opt.image.split('/')[-1].split('.')[0] + '.png'

        save_img_path = os.path.join(opt.results_path, save_img_name)
        cv2.imwrite(save_img_path, img_copy)

    print("Imapainting done", flush=True)

if __name__=="__main__":
    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
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

    #Images
    parser.add_argument('--image', type=str, default='')
    parser.add_argument('--mask', type=str, default='')

    opt = parser.parse_args()

    if opt.image == '' or opt.mask == '':
        raise ReferenceError("Wrong usage, use --image path/to/image --mask path/to/mask")

    print(opt, flush=True)

    if opt.gan_type == 'WGAN':
        impaint(opt)
