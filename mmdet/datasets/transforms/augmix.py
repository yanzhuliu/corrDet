import random
from .augmentations import augmentations, augmentations_all
import numpy as np
from PIL import Image
import cv2
from mmdet.registry import TRANSFORMS

@TRANSFORMS.register_module()
class AugMix:
    """
    AugMix transform. 
    Reference: https://github.com/google-research/augmix/blob/master/cifar.py
    """

    def __init__(self, all_ops=False, mixture_width=3, mixture_depth=-1, aug_severity=3, jsd_loss=False):
        self.aug_list = augmentations
        if all_ops:
            self.aug_list = augmentations_all
        self.mixture_width = mixture_width
        self.mixture_depth = mixture_depth
        self.aug_severity  = aug_severity
        self.jsd_loss      = jsd_loss
        if self.jsd_loss:
            raise NotImplementedError

    def __call__(self, results):
        ws = np.float32(np.random.dirichlet([1] * self.mixture_width))
        m  = np.float32(np.random.beta(1, 1))

        img= results['img'].astype(np.float32)
        mix= np.zeros_like(img)
        for i in range(self.mixture_width):
            image_aug = np.copy(img)
            depth     = self.mixture_depth if self.mixture_depth > 0 else np.random.randint(1, 4)
            for _ in range(depth):
                op    = np.random.choice(self.aug_list)
                image_aug = Image.fromarray(np.uint8(image_aug))
                image_aug = op(image_aug, self.aug_severity)
                image_aug = np.array(image_aug).astype(np.float32)
            mix += ws[i] * image_aug
        mixed = (1 - m)*img + m*mix
        #print(np.amin(img), np.amax(img))
        #print(np.amin(mixed), np.amax(mixed))
        #print('augment/augmix/'+results['filename'].split('/')[-1][:-4]+'_orig.jpg')
        #cv2.imwrite('augment/augmix/'+results['filename'][:-4]+'_orig.jpg', img)
        #cv2.imwrite('augment/augmix/'+results['filename'][:-4]+'_mixd.jpg', mixed)
        #cv2.imwrite('augment/augmix/'+results['filename'].split('/')[-1][:-4]+'_orig.jpg', img)
        #cv2.imwrite('augment/augmix/'+results['filename'].split('/')[-1][:-4]+'_mixd.jpg', mixed)
        results['img'] = mixed
        return results