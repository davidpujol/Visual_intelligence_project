import os
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

import DeepFillv2.utils as utils
from DeepFillv2.data import Data


def impaint(image, mask, use_gpu=True):

    # Choose the options for the inpainting
    opt = utils.init()

    # Load the model
    def load_model_generator(net, epoch, opt):
        model_name = 'deepfillv2_WGAN_G_epoch%d_batchsize%d.pth' % (epoch, 4)
        model_name = os.path.join('DeepFillv2/pretrained_model', model_name)
        pretrained_dict = torch.load(model_name, map_location=torch.device('cpu'))
        generator.load_state_dict(pretrained_dict)

    # ----------------------------------------
    #      Initialize training parameters
    # ----------------------------------------

    # Configure result path if not existing
    if not os.path.exists(opt.results_path):
        os.makedirs(opt.results_path)

    # Prepare the image and the mask from arrays
    cv2_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    cv2_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Build networks
    generator = utils.create_generator(opt).eval()
    print('-------------------------Loading Pretrained Model-------------------------')
    load_model_generator(generator, opt.epoch, opt)
    print('-------------------------Pretrained Model Loaded-------------------------')

    # To device
    if use_gpu:
        generator = generator.cuda()

    # ----------------------------------------
    #       Initialize training dataset
    # ----------------------------------------

    # Define the dataset
    dataset = Data(opt, cv2_img, cv2_mask)
    
    # Define the dataloader
    dataloader = DataLoader(dataset, batch_size = opt.batch_size, shuffle = False, pin_memory = True)#, num_workers = opt.num_workers)

    # For each images in the dataset do the inpainting
    for batch_id, (data_img, data_mask) in enumerate(dataloader):
        img = data_img
        mask = data_mask

        if use_gpu:
            img = img.cuda()
            mask = mask.cuda()

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

        utils.save_img(img_copy)

        print(img_copy)
        print("-------------------------Impainting done-------------------------", flush=True)

        return img_copy