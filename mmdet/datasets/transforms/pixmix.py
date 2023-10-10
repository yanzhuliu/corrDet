import os
#from mmdet.datasets import PIPELINES
from mmdet.registry import TRANSFORMS
import numpy as np
from PIL import Image
import cv2
from .augmentations import augmentations, augmentations_all, mixings
from mmengine.fileio import get
import mmcv
import glob
from random import randrange

#@PIPELINES.register_module()
@TRANSFORMS.register_module()
class PixMix:
    """
    PixMix transform. 
    Reference: https://github.com/andyzoujm/pixmix
    """

    def __init__(self, all_ops=False, mixture_depth=4, beta=3, aug_severity=3):
        self.aug_list = augmentations
        if all_ops:
            self.aug_list = augmentations_all

        self.mixture_depth = mixture_depth
        self.beta = beta
        self.aug_severity  = aug_severity
        self.color_type = 'color' #''unchanged'
        self.imdecode_backend = 'cv2'
        
        mixing_set_path = "/mnt/data/causal_rodc/data/fractals_and_fvis/images"
        mixing_list = glob.glob(mixing_set_path + "/*")
        mixing_list.sort()
        #print("mixing_list: ", len(mixing_list))
        self.mixing_set = mixing_list


    def __call__(self, results):
        
        #randomly select 1 image from mixing set
        rnd_idx = np.random.choice(len(self.mixing_set))
        mixing_filepath = self.mixing_set[rnd_idx]
        img_bytes = get(mixing_filepath)
        mixing_pic = mmcv.imfrombytes(img_bytes, flag=self.color_type, backend=self.imdecode_backend)
        mixing_pic = Image.fromarray(np.uint8(mixing_pic))

        #input image
        img = Image.fromarray(np.uint8(results['img']))
        
        #resize mixing_pic
        mix_width, mix_height = mixing_pic.size
        img_width, img_height = img.size
        scale = max(img_width/mix_width, img_height/mix_height) + 0.1
        
        new_size = (int(img_width*scale), int(mix_height*scale))
        new_mixing_pic = mixing_pic.resize(new_size)
            
        #crop mixing_pic
        x1, y1 = 0, 0
        if abs(mix_width - img_width) > 0:
            x1 = randrange(0, abs(mix_width - img_width))
        if abs(mix_height - img_height) > 0:
            y1 = randrange(0, abs(mix_height - img_height))
        cropped_img = new_mixing_pic.crop((x1, y1, x1 + img_width, y1 + img_height))
        mixing_pic = cropped_img

        mixings_fn = mixings
               
        if np.random.random() < 0.5:
            op = np.random.choice(self.aug_list)
            mixed = op(img.copy(), self.aug_severity)
        else:
            mixed = img.copy()
        mixed = np.array(mixed).astype(np.float32)
        
        for _ in range(np.random.randint(self.mixture_depth + 1)):
          
          if np.random.random() < 0.5:  
              op = np.random.choice(self.aug_list)
              aug_image_copy = op(img.copy(), self.aug_severity)
          else:
              aug_image_copy = mixing_pic.copy()

          mixed_op = np.random.choice(mixings_fn)
          aug_image_copy = np.array(aug_image_copy).astype(np.float32)
          mixed = mixed_op(mixed, aug_image_copy, self.beta)
          mixed = np.clip(mixed, 0, 255) #(0, 1)

        results['img'] = mixed
    
        #img = np.array(img).astype(np.float32)
        #cv2.imwrite('augment/pixmix/'+results['filename'].split('/')[-1][:-4]+'_orig.jpg', img)
        #cv2.imwrite('augment/pixmix/'+results['filename'].split('/')[-1][:-4]+'_mixd.jpg', mixed)

        return results