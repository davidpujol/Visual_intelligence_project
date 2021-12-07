import cv2
import numpy as np
import torch

from torch.utils.data import Dataset


class Data(Dataset):
    def __init__(self, opt, img, mask):
        self.opt = opt

        self.img = img
        self.mask = mask

    def __len__(self):
        return 1

    def __getitem__(self, index=0):
        # image
        img = self.img
        mask = self.mask[:, :, 0]

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
        mask = torch.from_numpy(mask.astype(np.float32) / 255.0).unsqueeze(0).contiguous()
        print(img.size(), flush=True)

        return img, mask
