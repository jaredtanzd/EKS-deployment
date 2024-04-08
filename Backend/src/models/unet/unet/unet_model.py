""" Full assembly of the parts to form the complete network """

#from .unet_parts import *
import pytorch_lightning as pl
from torch import Tensor
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import draw_segmentation_masks, make_grid
import wandb
import numpy as np


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)



class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)



class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_classes = n_classes
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits



def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        target = target.to(input.dtype)
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]


#def dice_coef(input, target):
#    assert input.size() == target.size()
#    smooth = 1.
#
#    iflat = input.view(-1)
#    tflat = target.view(-1)
#    intersection = (iflat * tflat).sum()
#        
#    return ((2. * intersection + smooth) /
#                          (iflat.sum() + tflat.sum() + smooth))
#

def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    for channel in range(input.shape[1]):
        dice += dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

    return dice / input.shape[1]




class Model(pl.LightningModule):
    '''
    n_classes is dropped here, it will be assigned at main.py according to dataloader  classes

    '''
    def __init__(self, in_channels, classes, bilinear=False, loss_config=None, learning_rate=None):
        super(Model, self).__init__()
        self.classes = ['background'] + [c.split('/')[-1] for c in classes] #background
        self.n_classes = len(self.classes)
        
        self.unet = UNet(in_channels, self.n_classes ) 

        self.learning_rate = learning_rate

        self.loss = eval(loss_config['target']) if loss_config['target'] is not None else torch.nn.CrossEntropyLoss()


    def forward(self, input):
        out = self.unet(input)
        return out

    def training_step(self, batch, batch_idx):
        x,y = batch
        logits = self(x)
        pred = torch.nn.functional.softmax(logits,dim=1)
        y = rearrange(y, 'b h w c -> b c h w')
        loss = self.loss(pred, y)
        self.log('Loss',loss)
        return loss

    def training_step_end(self, training_step_outputs):
        '''If need to do sth on the step train output'''
        pass
        #gpu_0_pred = training_step_outputs[0]["pred"]
        #gpu_1_pred = training_step_outputs[1]["pred"]
        #gpu_n_pred = training_step_outputs[n]["pred"]
        ## this softmax now uses the full batch
        #loss = nce_loss([gpu_0_pred, gpu_1_pred, gpu_n_pred])
        #return loss

    def validation_step(self, batch, batch_idx):
        x,y = batch
        #y = rearrange(y, 'b h w c -> b c h w')
        logits = self(x)
        pred = torch.argmax(torch.nn.functional.softmax(logits,dim=1), dim=1, keepdims=True)
        pred = torch.zeros_like(logits).scatter_(1, pred,1)
        dice = multiclass_dice_coeff(pred, y)
        self.log('Metric/Dice',dice , on_epoch=True)
        return {'input':x, 'prediction':pred, 'gt':y}
    
    def validation_epoch_end(self, validation_step_outputs):

        '''
        Currently will only take the first batch to draw grid.
        '''
        inp , pred, y = validation_step_outputs[0]['input'] , validation_step_outputs[0]['prediction'] , validation_step_outputs[0]['gt']
        #max_show = min(4, inp.shape[0])
        #grid = [self.draw_mask(inp[b], pred[b]) for b in range(max_show)]
        #grid = make_grid(grid, nrow= len(grid)//2).float()
        #self.logger.experiment.log({'Validation plot' : wandb.Image(grid)})
        max_show = min(3, inp.shape[0])

        mask_list = []
        for b in range(max_show):
            im = inp[b].repeat(3,1,1).permute(1,2,0).cpu().numpy()
            class_labels = dict(zip(np.arange(len(self.classes)),self.classes))
            mask_img = wandb.Image(im, masks={
                  "predictions": {
                          "mask_data": pred[b].argmax(0).cpu().numpy(),
                              "class_labels": class_labels
                                },
                    "ground_truth": {
                          "mask_data": y[b].argmax(0).cpu().numpy(),
                              "class_labels": class_labels
                              },
                                })
            mask_list += [mask_img]

        self.logger.experiment.log({'Validation plot' : mask_list})



    
    def draw_mask(self, input: torch.Tensor, pred : torch.Tensor):
        '''
        input (tensor) : [ 1, H, W]
        pred (tensor) : [C , H, W]
        '''
        cl =  ['black','red','green','blue','yellow']
        im = (input.repeat(3,1,1) * 255).to(torch.uint8)
        return draw_segmentation_masks(im.cpu()  , pred.to(bool).cpu() ,colors= cl[:pred.shape[0]], alpha=0.4)



    def configure_optimizers(self):
        lr = self.learning_rate
        opt = torch.optim.SGD(self.unet.parameters(),
                                 lr=0.0001, momentum=0.99, nesterov=True)
        #opt = torch.optim.Adam(self.unet.parameters(),
        #                         lr=lr, betas=(0.5, 0.9))
        return opt
