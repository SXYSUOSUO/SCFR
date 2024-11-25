# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import random

import torch
import torchvision
from torchvision.transforms import functional as F
import pdb


from PIL import ImageFilter, ImageOps


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, image,target):
        do_it = random.random() <= self.prob
        if not do_it:
            return image,target

        return image.filter(ImageFilter.GaussianBlur(radius=random.uniform(self.radius_min, self.radius_max))),target
        


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            return ImageOps.solarize(image),target
        else:
            return image,target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Resize(object):
    def __init__(self, min_size, max_size):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image, target):
        size = self.get_size(image.size)
        image = F.resize(image, size)
        target = target.resize(image.size)
        #print (size)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.hflip(image)
            target = target.transpose(0)
            
        return image, target

class RandomVerticalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.vflip(image)
            target = target.transpose(1)
        return image, target


def random_crop(image, min_ratio=0.8, max_ratio=1.0):

    w, h = image.size
    #print (w,h,'111')
    
    ratio = random.random()
    
    scale = min_ratio + ratio * (max_ratio - min_ratio)
    
    new_h = int(h*scale)    
    new_w = int(w*scale)
    
    y = random.randint(0, h - new_h)    
    x = random.randint(0, w - new_w)
    
    image = F.crop(image,y,x,new_h,new_w)
    #image = F.crop(image,x,y,new_w,new_h)
    #print (image.size)
    #print (ratio)
    #pdb.set_trace()
    return image, [x,y,x+new_w,y+new_h]




class RandomCrop(object):
    def __init__(self,scale):
        self.scale = scale

    def __call__(self, image, target):
        w,h = target.size
        new_image, box = random_crop(image,min_ratio=self.scale)
        target_new = target.crop(box)
        if len(target_new.bbox) > 0 :
            image = new_image   
            target = target_new
            #print (image.size)
        return image, target





class ColorJitter(object):
    def __init__(self,prob,
                 brightness=None,
                 contrast=None,
                 saturation=None,
                 hue=None,
                 ):
        
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue)
        
        self.prob = prob
        #self.color_jitter_b= torchvision.transforms.ColorJitter(brightness=brightness)
        #self.color_jitter_c = torchvision.transforms.ColorJitter(contrast=contrast)
        #self.color_jitter_s = torchvision.transforms.ColorJitter(saturation=saturation)
        #self.color_jitter_h = torchvision.transforms.ColorJitter(hue=hue)

    def __call__(self, image, target):
        if random.random() < self.prob:
          image = self.color_jitter(image)
            
        #if random.random() < self.prob:
        #    image = self.color_jitter_s(image)
        #if random.random() < self.prob:
        #    image = self.color_jitter_s(image)
        #if random.random() < self.prob:
        #    image = self.color_jitter_h(image)
        return image, target




class ToTensor(object):
    def __call__(self, image, target):
        return F.to_tensor(image), target


class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):  
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, target):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target
