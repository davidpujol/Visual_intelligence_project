import cv2
import numpy as np
import torch

from torch.utils.data import Dataset


class Data(Dataset):
    def __init__(self, opt):
        self.opt = opt

        self.img = opt.image
        self.mask = opt.mask

    def __len__(self):
        return 1

    def __getitem__(self, index=0):
        # image
        img = cv2.imread(self.img)
        mask = cv2.imread(self.mask)[:, :, 0]

        # find the Minimum bounding rectangle in the mask
        '''
        contours, hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cidx, cnt in enumerate(contours):
            (x, y, w, h) = cv2.boundingRect(cnt)
            mask[y:y+h, x:x+w] = 255
        '''
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
        mask = torch.from_numpy(mask.astype(np.float32) / 255.0).unsqueeze(0).contiguous()
        print(img.size(), flush=True)
        return img, mask
