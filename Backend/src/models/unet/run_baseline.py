import os
import logging
from pathlib import Path

import yaml
with open(Path(__file__).parent / "backend_unet.yaml", 'r') as f:
    valuesYaml = yaml.load(f, Loader=yaml.FullLoader)

os.environ['CUDA_VISIBLE_DEVICES'] = str(valuesYaml['CUDA_VISIBLE_DEVICES'])

import imageio
from PIL import Image

import numpy as np

import torch
import torch.nn.functional as F
from torchvision import transforms


from unet import UNet
from unet_utils import plot_img_and_mask
from unet_utils import BasicDataset

from torchvision.utils import save_image


MODEL_DIRECTORY_PREFIX = str(Path(__file__).parent) + '/Pytorch-UNet-master/' # or without code
#TARGET_DIRECTORY = '/data/data_repo/neuro_img/anat_brain_img/cache/baseline/'

import sys
sys.path.insert(0, Path(__file__).parent.parent)
from src.awsclient import Bucket

class unet:
    
    def __init__(self, cfg, task): # cfg is from the 2nd yaml
        
        self.cfg = {}
        
        self.cuda = True if torch.cuda.is_available() else False
        
        self.cfg['n_epochs'] = cfg['baseline']['n_epochs']
        net = UNet(n_channels=1, n_classes=2)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_type = 'checkpoints_' + "gm"  + '_alt'
        model_epoch = 'checkpoint_epoch' + str(self.cfg['n_epochs']) + '.pth'
        path_to_model = MODEL_DIRECTORY_PREFIX + model_type + '/' + model_epoch
 
        if not os.path.exists(path_to_model):
            bucket = Bucket(path_to_model)
            bucket.exists(model_type + '/' + model_epoch, get=True)

        logging.info(f'Loading model {path_to_model}')

        net.to(device=self.device)
        net.load_state_dict(torch.load(path_to_model, map_location=self.device))
        self.net = net

        
    def __call__(self, data):

        # os.makedirs('./data/mask_' + args.task, exist_ok=True)
        # out_files = get_output_filenames(args)

        #logging.info('Model loaded!')
        
        #os.makedirs(TARGET_DIRECTORY + args.get('task') + '/', exist_ok=True)

        #for i, dcmfile in enumerate(data):
            
        #logging.info(f'\nPerforming Tissue Segmentation  ...\n')

        #img = Image.fromarray(np.array(dcmfile.im))

        if not all([isinstance(i, (np.ndarray, torch.Tensor)) for i in data ]):
            img = list(map(lambda x: x.im[np.newaxis,...] , data))
            img = np.concatenate(img, axis=0)
            instance_id = list(map(lambda x: x.orthancID, data))
    
        

            mask = predict_img(net=self.net,
                                   full_img=img,
                                   device=self.device)



            return dict(zip(instance_id, mask))
        else:
            img = data
            mask = predict_img(net=self.net,
                                   full_img=img,
                                   device=self.device)

            return mask

            
            #print(type(mask), mask.shape)
            
            # save_image(mask, TARGET_DIRECTORY + args.get('task') + '/' + f'{args.get("instance")}.png', nrow=1, normalize=False)

#             if not args.no_save:
            #out_filename = TARGET_DIRECTORY + args.get('task') + '/' + f'{args.get("instance")}.png'
            #result = mask_to_image(mask)
            #result.save(out_filename)
            #logging.info(f'Mask saved to {out_filename}')
                
            #print(i, 'done')


def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()


    #full_img is now a numpy [B,H,W]
    #img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor, is_mask=False))

    if isinstance(full_img, np.ndarray):
        img = torch.from_numpy(full_img)
        img = img.to(torch.float32) / np.ptp(img)
        img = img.unsqueeze(1)
        img = img.to(device=device, dtype=torch.float32)

    else: img = full_img.to(device=device, dtype=torch.float32)

    limit_batch = 100
    with torch.no_grad():
        if img.shape[0] > limit_batch:
            output = []
            for i in range(int(np.ceil(img.shape[0] / limit_batch))):
                output += [net(img[i*limit_batch:(i+1)*limit_batch]).cpu()]
            output = torch.cat(output)
        else:
            output = net(img)
        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)[:,1,...]
        else:
            probs = torch.sigmoid(output)[:,1,...]

        #tf = transforms.Compose([
        #    transforms.ToPILImage(),
        #    transforms.Resize((full_img.size[1], full_img.size[0])),
        #    transforms.ToTensor()
        #])

        #full_mask = tf(probs.cpu()).squeeze()
    full_mask = probs.cpu().squeeze(1)
    torch.cuda.empty_cache()
    #if net.n_classes == 1:
    return (full_mask > out_threshold).numpy()
    #else:
    #    return F.one_hot(full_mask.argmax(dim=0), net.n_classes).permute(2, 0, 1).numpy()

def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))

