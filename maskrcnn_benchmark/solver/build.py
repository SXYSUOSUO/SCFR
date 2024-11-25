# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from .lr_scheduler import WarmupMultiStepLR
from timm.scheduler.cosine_lr import CosineLRScheduler

def make_optimizer(cfg, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    if cfg.SOLVER.OPTIMIZER =="SGD":
        optimizer = torch.optim.SGD(params, lr, momentum=cfg.SOLVER.MOMENTUM)
    elif cfg.SOLVER.OPTIMIZER =="Adam":
        optimizer = torch.optim.Adam(params, lr)
    return optimizer


def make_lr_scheduler(cfg, optimizer):
  
    #return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
    
    return CosineLRScheduler(optimizer, 
                             t_initial = cfg.SOLVER.ITERS, 
                             #cycle_mul=1.0, 
                             #t_mul = 1.0,
                             lr_min = cfg.SOLVER.BASE_LR*0.01, 
                             #cycle_decay = cfg.SOLVER.DECAY_RATE,
                             # decay_rate =  cfg.SOLVER.DECAY_RATE,
                             warmup_lr_init = cfg.SOLVER.BASE_LR*0.1,
                             warmup_t = cfg.SOLVER.WARMUP_ITERS, 
                             cycle_limit=8,
                             t_in_epochs=False,
                            )
    '''
    return WarmupMultiStepLR(
        optimizer,
        cfg.SOLVER.STEPS,
        cfg.SOLVER.GAMMA,
        warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
        warmup_iters=cfg.SOLVER.WARMUP_ITERS,
        warmup_method=cfg.SOLVER.WARMUP_METHOD,)
        '''
