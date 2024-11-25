# ---------------------------------------------------------------------------
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

# from torch.nn import DataParallel  # or your customized DataParallel module
# from sync_batchnorm import SynchronizedBatchNorm1d, patch_replication_callback
# ‘_a’ means all


from maskrcnn_benchmark.modeling.swin_transformer import Swin_transformer
from . import fpt3 as fpt_module
# import maskrcnn_benchmark.nn as mynn
import pdb


def group_norm(num_channels):
    return nn.GroupNorm(8, num_channels)

class CFPT(nn.Module):
    def __init__(self, feature_dim,window_size,top_blocks =None, with_norm ='none'):
        super(CFPT, self).__init__()
        self.feature_dim = feature_dim
        #assert upsample_method in ['nearest', 'bilinear']

        self.top_blocks = top_blocks
        self.with_norm = "none"
        
        self.window_size = window_size
        #self.fpn_upsample = interpolate
        
        assert with_norm in ['group_norm', 'batch_norm', 'none','layer_norm']
        if with_norm == 'batch_norm':
            norm = nn.BatchNorm2d
        elif with_norm == 'group_norm':
            norm = group_norm
        elif with_norm == 'layer_norm':
            norm = nn.LayerNorm
        #self.norm = norm(feature_dim)
        #pdb.set_trace()
        if self.with_norm!="none":
            self.norm2 = [norm(feature_dim).cuda(),norm(feature_dim).cuda(),norm(feature_dim).cuda(),norm(feature_dim).cuda()]
            self.norm1 = [norm(feature_dim).cuda(),norm(feature_dim).cuda(),norm(feature_dim).cuda(),norm(feature_dim).cuda()]



        
        '''
        self.cts1 = [LocalityconstrainedGT(feature_dim, feature_dim, kernel_size=3, padding=1,stride=1, num_heads=1).cuda(),
        LocalityconstrainedGT(feature_dim, feature_dim, kernel_size=3, padding=1,stride=1, num_heads=1).cuda(),
        LocalityconstrainedGT(feature_dim, feature_dim, kernel_size=3, padding=1,stride=1, num_heads=1).cuda(),
        LocalityconstrainedGT(feature_dim, feature_dim, kernel_size=3, padding=1,stride=1, num_heads=1).cuda()]

        self.cts2 = [LocalityconstrainedGT(feature_dim, feature_dim, kernel_size=3, padding=1,stride=1, num_heads=1).cuda(),
        LocalityconstrainedGT(feature_dim, feature_dim, kernel_size=3, padding=1,stride=1, num_heads=1).cuda(),
        LocalityconstrainedGT(feature_dim, feature_dim, kernel_size=3, padding=1,stride=1, num_heads=1).cuda(),
        LocalityconstrainedGT(feature_dim, feature_dim, kernel_size=3, padding=1,stride=1, num_heads=1).cuda()]

        '''
        self.cts1 = [
        Swin_transformer(in_channels=feature_dim, out_channels=feature_dim, window_size=window_size, num_heads=2, position_emd=False, bias=False).cuda(),
        Swin_transformer(in_channels=feature_dim, out_channels=feature_dim, window_size=window_size, num_heads=2, position_emd=False, bias=False).cuda(),
        Swin_transformer(in_channels=feature_dim, out_channels=feature_dim, window_size=window_size, num_heads=2, position_emd=False, bias=False).cuda(),
        Swin_transformer(in_channels=feature_dim, out_channels=feature_dim, window_size=window_size, num_heads=2, position_emd=False, bias=False).cuda()]

        self.cts2 = [
        Swin_transformer(in_channels=feature_dim, out_channels=feature_dim, window_size=window_size, num_heads=2, position_emd=False, bias=False).cuda(),
        Swin_transformer(in_channels=feature_dim, out_channels=feature_dim, window_size=window_size, num_heads=2, position_emd=False, bias=False).cuda(),
        Swin_transformer(in_channels=feature_dim, out_channels=feature_dim, window_size=window_size, num_heads=2, position_emd=False, bias=False).cuda(),
        Swin_transformer(in_channels=feature_dim, out_channels=feature_dim, window_size=window_size, num_heads=2, position_emd=False, bias=False).cuda()]
        
        '''
        if with_norm != 'none':
            self.fpts = [ nn.Sequential(*[nn.Conv2d(feature_dim * 2, feature_dim, 1, padding=0, bias=False), norm(feature_dim)]).cuda(),
                               nn.Sequential(*[nn.Conv2d(feature_dim * 2, feature_dim, 1, padding=0, bias=False), norm(feature_dim)]).cuda(),
                               nn.Sequential(*[nn.Conv2d(feature_dim * 2, feature_dim, 1, padding=0, bias=False), norm(feature_dim)]).cuda(),
                               nn.Sequential(*[nn.Conv2d(feature_dim * 2, feature_dim, 1, padding=0, bias=False), norm(feature_dim)]).cuda()]
            
        else:
            self.fpts =[ nn.Conv2d(feature_dim * 2, feature_dim, 1, padding=0).cuda(),
                             nn.Conv2d(feature_dim * 2, feature_dim, 1, padding=0).cuda(),
                             nn.Conv2d(feature_dim * 2, feature_dim, 1, padding=0).cuda(),
                             nn.Conv2d(feature_dim * 2, feature_dim, 1, padding=0).cuda()]
        '''
      
       

    def forward(self, features):  # 输入：fpn的2-5个阶段的特征图


        l=int(features[0].shape[0]/2)
        features1=[]
        features2=[]
        
        for i in range(0,len(features)):
            features1.append(features[i][0:l])
            features2.append(features[i][l:])
        
        for i in range(0,len(features1)):
            #db.set_trace()
            if self.with_norm == "none":
                features1[i] = F.relu(features1[i] + self.cts1[i](features1[i],features2[i])) 
                features2[i] = F.relu(features2[i] + self.cts2[i](features2[i],features1[i])) 
            else:
                features1[i] = F.relu(features1[i] + self.norm1[i](self.cts1[i](features1[i],features2[i]))) 
                features2[i] = F.relu(features2[i] + self.norm2[i](self.cts2[i](features2[i],features1[i]))) 
            
            #features1[i] =  self.fpts[i](t1)
            #features2[i] =  self.fpts[i](t2)

        cfpt_features = []
        for i in range(0,len(features1)):
            #pdb.set_trace()
            cfpt_features.append(torch.cat((features1[i],features2[i]),0))
  
        if isinstance(self.top_blocks, fpt_module.LastLevelP6P7) or isinstance(self.top_blocks, fpt_module.LastLevelMaxPool):
            #pdb.set_trace()
            last_results = self.top_blocks(cfpt_features[-1])
            cfpt_features.extend(last_results)
        #pdb.set_trace()
        return cfpt_features  # 输出：经过fpt之后的对应2-5特征图
      

