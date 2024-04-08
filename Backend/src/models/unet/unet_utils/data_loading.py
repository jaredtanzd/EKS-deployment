import logging
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

import imageio


class BasicDataset(Dataset):
    
    def __init__(self, images_dir: str, masks_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
            
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(pil_img, scale, is_mask):
        
        # w, h = pil_img.size
        # newW, newH = int(scale * w), int(scale * h)
        # assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        # pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        
        img_ndarray = np.array(pil_img) # asarray

        if not is_mask:
            
            if img_ndarray.ndim == 2: # single channel
                img_ndarray = img_ndarray[np.newaxis, ...]
            else: # channel, shift to pytorch convention
                img_ndarray = img_ndarray.transpose((2, 0, 1))
                
            use_int = False
            if use_int:
                assert img_ndarray.max() <= 1
                img_ndarray *= 255
                img_ndarray = img_ndarray.astype(np.uint8)
                # if on, this will lose some info. but apparently it works better!?
                # from min-max output, cast to 0-255, int so lose info, then /255 to make 0-1...
                
            if img_ndarray.max() > 1:
                print('img_ndarray.max()', img_ndarray.max())
                print('use_int', use_int, "; ct img values bigger than 1")
                img_ndarray = img_ndarray / 255

        return img_ndarray

    @staticmethod
    def load(filename):
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        elif ext in ['.tiff']:
            return Image.fromarray(np.array(imageio.imread(filename))) # ct            
        else: # png, mask
            return Image.open(filename)

    def __getitem__(self, idx):
        
        name = self.ids[idx]
        # print(self.masks_dir)
        # print(self.images_dir)
        mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = self.load(mask_file[0])
        img = self.load(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, masks_dir, scale=1):
        super().__init__(images_dir, masks_dir, scale, mask_suffix='_mask')
