# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from . import transforms as T


def build_transforms(cfg, is_train=True):

    to_bgr255 = cfg.INPUT.TO_BGR255
    normalize_transform = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=to_bgr255)
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        flip_prob = 0.5  # cfg.INPUT.FLIP_PROB_TRAIN
        brightness =0.4
        contrast = 0.4
        saturation = 0.3
        hue = 0.2
        #color_jitter1 = T.ColorJitter(brightness = brightness,contrast=0,saturation=0,hue=0)
        #color_jitter2 = T.ColorJitter(brightness=0, contrast = contrast,saturation=0,hue=0 )
        #color_jitter3 = T.ColorJitter(brightness=0,contrast=0,saturation = saturation,hue=0)
        #color_jitter4 = T.ColorJitter(brightness=0,contrast=0,saturation=0,hue = hue)
        
        color_jitter = T.ColorJitter(
        prob=cfg.DATASETS.JITPROB,
        brightness = brightness,
        contrast = contrast,
        saturation = saturation,
        hue = hue,
        )
        
        transform = T.Compose([
            T.RandomCrop(cfg.INPUT.SCALE),
            color_jitter,
            #color_jitter2,
            #color_jitter3,
            #color_jitter4,
            T.Resize(min_size, max_size),
            T.RandomHorizontalFlip(flip_prob),
            T.ToTensor(),
            normalize_transform,
        ])
    
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        transform = T.Compose(
        [

            T.Resize(min_size, max_size),
            T.ToTensor(),
            normalize_transform,
        ]
        )
        

    return transform


