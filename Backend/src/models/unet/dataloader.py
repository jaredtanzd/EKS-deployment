

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
import os
import imageio
import numpy as np
from PIL import Image
import cv2
from einops import rearrange

class LoadTissue:
    def __init__(self,input, label, size, augment=True ,**kwargs):
        #self.data_cfg = config
        self.input_path = input
        self.class_target = label
        self.augment = augment
        self.input_images = os.listdir( self.input_path )
        self.img_size = size

        self.transform_labeled = A.Compose([
 #                         A.ToGray(p=0.5),
                          A.RandomGamma(p=0.3),
                          A.HorizontalFlip(p=0.5),
                          A.augmentations.geometric.transforms.Affine(scale=1.2,rotate=(-15,15),shear=(15,15),),
                          A.RandomResizedCrop(height=size,
                                            width=size,
                                            scale=(0.8,1.0),
                                            p=1),
                          ToTensorV2(),])

    def get_data(self, name):
        x = np.array(imageio.imread(Path(self.input_path) / name))
        path_y = [np.zeros_like(x)[..., np.newaxis] ] #background class
        for c in self.class_target:
            path_y += [ np.array(Image.open(Path(c) / str(name.split('.')[0] + '.png' ) ))[ ..., np.newaxis] ]

        mask = np.concatenate(path_y, -1)
        h0, w0 = x.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # ratio
        if (r != 1) or (h0 != w0):  # if sizes are not equal
            interp = cv2.INTER_LINEAR if self.augment else cv2.INTER_AREA  # random.choice(self.rand_interp_methods)
            x = cv2.resize(x, (self.img_size, self.img_size), interpolation=interp)
            mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=interp)

        #data is flipped
        x = np.flip(rearrange(x, 'h w -> w h'), axis=-2).copy() # image is 2D
        mask = np.flip(rearrange(mask,'h w c -> w h c' ), axis=0).copy() # channel at last dim

        return x, mask



        
        
    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        im , mask = self.get_data(self.input_images[idx])
        if self.augment:
            augmented = self.transform_labeled(image=im, mask=mask)
            return augmented['image'].to(torch.float), augmented['mask'].to(torch.float)

        else:
            return torch.from_numpy(im).to(torch.float)[None,], rearrange(torch.from_numpy(mask.copy()), 'h w c -> c h w')
