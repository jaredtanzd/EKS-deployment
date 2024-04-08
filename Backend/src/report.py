
import numpy as np
import cv2
from typing import Tuple
from pathlib import Path
from PIL import Image



def report_vis( im_path:str , mask_path:str , color : Tuple[int,int,int] = (255 ,0 ,0) ):
    img = np.array(Image.open(im_path))
    msk = np.load(mask_path) #a numpy file
   
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB).transpose(2,0,1)
    #msk = msk[:,:,0]
    #msk[msk == 30] = 0
    
    new_path = Path(mask_path).parent / (Path(mask_path).stem + '_report.png' )

    
    combined = overlay(img, msk ,color , resize=None).transpose(1,2,0)

    Image.fromarray(combined).save(new_path)

    return new_path
    





def overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.5, 
    resize: Tuple[int, int] = (1024, 1024)
) -> np.ndarray:
    """Combines image and its segmentation mask into a single image.
    
    Params:
        image: Training image.
        mask: Segmentation mask.
        color: Color for segmentation mask rendering.
        alpha: Segmentation mask's transparency.
        resize: If provided, both image and its mask are resized before blending them together.
    
    Returns:
        image_combined: The combined image.
        
    """
    color = np.asarray(color).reshape(3, 1, 1)
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()
    
    if resize is not None:
        image = cv2.resize(image.transpose(1, 2, 0), resize)
        image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)
    
    image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)
    
    return image_combined
