# ---------------------------------------------------------------------------
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

# from torch.nn import DataParallel  # or your customized DataParallel module
# from sync_batchnorm import SynchronizedBatchNorm1d, patch_replication_callback
# ‘_a’ means all

from maskrcnn_benchmark.modeling.swin_transformer import Swin_transformer, Swin_transformer_up
# import maskrcnn_benchmark.nn as mynn
import pdb




class FPT(nn.Module):
    def __init__(self, feature_dim, window_size, share_fpt = True, position_emd = False, need_head=True,top_blocks =None, with_norm='none'):
        super(FPT, self).__init__()
        
        self.fpt1 = FPT_B(feature_dim, window_size, position_emd, need_head, top_blocks, with_norm)
        self.fpt2 = FPT_B(feature_dim, window_size, position_emd, need_head, top_blocks, with_norm)
        self.share_fpt = share_fpt

    def forward(self, features): 
      
        if self.share_fpt == False:
            l=int(features[0].shape[0]/2)
            features1=[]
            features2=[]
        
            for i in range(0,len(features)):
                features1.append(features[i][0:l])
                features2.append(features[i][l:])
        
            features1=self.fpt1(features1)
            features2=self.fpt2(features2)
        
            fpt_features = []
            for i in range(0,len(features1)):
                #pdb.set_trace()
                fpt_features.append(torch.cat((features1[i],features2[i]),0))
        else:
            fpt_features = self.fpt1(features)
        #pdb.set_trace()
        if isinstance(self.fpt1.top_blocks, LastLevelP6P7) or isinstance(self.fpt1.top_blocks, LastLevelMaxPool):
            last_results = self.top_blocks(fpt_features[-1])
            fpt_features.extend(last_results)
        
        return fpt_features  # 输出：经过fpt之后的对应2-5特征图



def group_norm(num_channels):
    return nn.GroupNorm(8, num_channels)

class FPT_B(nn.Module):
    def __init__(self, feature_dim, window_size, position_emd = False, need_head=True,top_blocks =None, with_norm='none'):
        super(FPT_B, self).__init__()
        self.feature_dim = feature_dim
        #assert upsample_method in ['nearest', 'bilinear']
        self.need_head = need_head

        self.top_blocks = top_blocks
        
        #self.fpn_upsample = interpolate
        self.window_size = window_size

        
        assert with_norm in ['group_norm', 'batch_norm', 'none', 'layer_norm']
        if with_norm == 'batch_norm':
            norm = nn.BatchNorm2d
        elif with_norm == 'group_norm':
            norm = group_norm
        elif with_norm == 'layer_norm':
            norm = nn.LayerNorm
        

        self.st_p5 = Swin_transformer(in_channels=feature_dim, out_channels=feature_dim, window_size=5, num_heads=2, position_emd = position_emd, bias=False)
        self.st_p4 = Swin_transformer(in_channels=feature_dim, out_channels=feature_dim, window_size=5, num_heads=2, position_emd = position_emd, bias=False)
        self.st_p3 = Swin_transformer(in_channels=feature_dim, out_channels=feature_dim, window_size=5, num_heads=2, position_emd = position_emd, bias=False)
        
        self.gt_p4_p5 = Swin_transformer_up(in_channels=feature_dim, out_channels=feature_dim, window_size=5, scale=2, num_heads=2, position_emd=False, bias=False)
        self.gt_p3_p4 = Swin_transformer_up(in_channels=feature_dim, out_channels=feature_dim, window_size=5, scale=2, num_heads=2, position_emd=False, bias=False)
        self.gt_p3_p5 = Swin_transformer_up(in_channels=feature_dim, out_channels=feature_dim, window_size=5, scale=4, num_heads=2, position_emd=False, bias=False)
        
        

        if with_norm != 'none':
            if need_head ==True:
                self.fpn_p5_1x1 = nn.Sequential(*[nn.Conv2d(2048, feature_dim, 1, bias=False), norm(feature_dim)])
                self.fpn_p4_1x1 = nn.Sequential(*[nn.Conv2d(1024, feature_dim, 1, bias=False), norm(feature_dim)])
                self.fpn_p3_1x1 = nn.Sequential(*[nn.Conv2d(512, feature_dim, 1, bias=False), norm(feature_dim)])
                
                

            self.fpt_p5 = nn.Sequential(
                *[nn.Conv2d(feature_dim * 2, feature_dim, 1, padding=0, bias=False), norm(feature_dim)])
            self.fpt_p4 = nn.Sequential(
                *[nn.Conv2d(feature_dim * 3, feature_dim, 1, padding=0, bias=False), norm(feature_dim)])
            self.fpt_p3 = nn.Sequential(
                *[nn.Conv2d(feature_dim * 4, feature_dim, 1, padding=0, bias=False), norm(feature_dim)])
          
            

        else:
            if need_head ==True:
                self.fpn_p5_1x1 = nn.Conv2d(2048, feature_dim, 1)
                self.fpn_p4_1x1 = nn.Conv2d(1024, feature_dim, 1)
                self.fpn_p3_1x1 = nn.Conv2d(512, feature_dim, 1)
                self.fpn_p2_1x1 = nn.Conv2d(256, feature_dim, 1)

            self.fpt_p5 = nn.Conv2d(feature_dim * 2, feature_dim, 1, padding=0)
            self.fpt_p4 = nn.Conv2d(feature_dim * 3, feature_dim, 1, padding=0)
            self.fpt_p3 = nn.Conv2d(feature_dim * 4, feature_dim, 1, padding=0)
            self.fpt_p2 = nn.Conv2d(feature_dim * 5, feature_dim, 1, padding=0)
            
  

        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data, a=1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, features):  # 输入：fpn的2-5个阶段的特征图
        #res2 = res_feature0]
        
        '''
        l=int(features[0].shape[0]/2)
        features1=[]
        features2=[]
        
        for i in range(0,len(features)):
            features1.append(features[i][0:l])
            features2.append(features[i][l:])
        '''
        if self.need_head ==True:
            fpn_p5_1 = self.fpn_p5_1x1(features[3])
            fpn_p4_1 = self.fpn_p4_1x1(features[2])
            fpn_p3_1 = self.fpn_p3_1x1(features[1])
            

                
        else:
            fpn_p5_1 = features[-1]
            fpn_p4_1 = features[-2]
            fpn_p3_1 = features[-3]
           
            

            #fpn_p2_1 = res2
        # print(fpn_p5_1.shape,fpn_p4_1.shape,fpn_p3_1.shape,fpn_p2_1.shape)
        #pdb.set_trace()
        
        fpt_p5_out = torch.cat([self.st_p5(fpn_p5_1,fpn_p5_1), fpn_p5_1], 1)
        fpt_p4_out = torch.cat([self.st_p4(fpn_p4_1,fpn_p4_1), self.gt_p4_p5(fpn_p4_1, fpn_p5_1), fpn_p4_1], 1)
        fpt_p3_out = torch.cat([self.st_p3(fpn_p3_1,fpn_p3_1), self.gt_p3_p5(fpn_p3_1, fpn_p5_1), self.gt_p3_p4(fpn_p3_1, fpn_p4_1), fpn_p3_1], 1)
        
        
        #pdb.set_trace()
        fpt_p5 = F.gelu(self.fpt_p5(fpt_p5_out))
        fpt_p4 = F.gelu(self.fpt_p4(fpt_p4_out))
        fpt_p3 = F.gelu(self.fpt_p3(fpt_p3_out)) 
   

        #fpt_p5_out = torch.stack((self.st_p5(fpn_p5_1,fpn_p5_1), self.rt_p5_p4(fpn_p5_1, fpn_p4_1),
        #                        self.rt_p5_p3(fpn_p5_1, fpn_p3_1), self.rt_p5_p2(fpn_p5_1, fpn_p2_1), fpn_p5_1), 1)
        #fpt_p5 = F.relu(fpt_p5_out.sum(dim=1))
        #fpt_p5 =F.relu(self.st_p5(fpn_p5_1,fpn_p5_1)+self.rt_p5_p4(fpn_p5_1, fpn_p4_1)+self.rt_p5_p3(fpn_p5_1, fpn_p3_1)+self.rt_p5_p2(fpn_p5_1, fpn_p2_1)+fpn_p5_1)
        #fpt_p4_out = torch.stack((self.st_p4(fpn_p4_1,fpn_p4_1), self.rt_p4_p3(fpn_p4_1, fpn_p3_1),
        #                        self.rt_p4_p2(fpn_p4_1, fpn_p2_1), self.gt_p4_p5(fpn_p4_1, fpn_p5_1), fpn_p4_1), 1)
        #fpt_p4 = F.relu(fpt_p4_out.sum(dim=1))
        #fpt_p4 = F.relu(self.st_p4(fpn_p4_1,fpn_p4_1)+self.rt_p4_p3(fpn_p4_1, fpn_p3_1)+self.rt_p4_p2(fpn_p4_1, fpn_p2_1)+self.gt_p4_p5(fpn_p4_1, fpn_p5_1)+fpn_p4_1)
        #fpt_p3_out = torch.stack((self.st_p3(fpn_p3_1,fpn_p3_1), self.rt_p3_p2(fpn_p3_1, fpn_p2_1),
        #                        self.gt_p3_p4(fpn_p3_1, fpn_p4_1), self.gt_p3_p5(fpn_p3_1, fpn_p5_1), fpn_p3_1), 1)
        #fpt_p3 = F.relu(fpt_p3_out.sum(dim=1))
        #fpt_p3 = F.relu(self.st_p3(fpn_p3_1,fpn_p3_1)+self.rt_p3_p2(fpn_p3_1, fpn_p2_1)+self.gt_p3_p4(fpn_p3_1, fpn_p4_1)+self.gt_p3_p5(fpn_p3_1, fpn_p5_1)+fpn_p3_1)
        #fpt_p2_out = torch.stack((self.st_p2(fpn_p2_1,fpn_p2_1), self.gt_p2_p3(fpn_p2_1, fpn_p3_1),
         #                       self.gt_p2_p4(fpn_p2_1, fpn_p4_1), self.gt_p2_p5(fpn_p2_1, fpn_p5_1), fpn_p2_1), 1)
        #fpt_p2 = F.relu(fpt_p2_out.sum(dim=1))
        #fpt_p2 = F.relu(self.st_p2(fpn_p2_1,fpn_p2_1)+self.gt_p2_p3(fpn_p2_1, fpn_p3_1)+self.gt_p2_p4(fpn_p2_1, fpn_p4_1)+self.gt_p2_p5(fpn_p2_1, fpn_p5_1)+fpn_p2_1)

        # fpt_p5= self.fpt_p5(fpt_p5_out)
        #fpt_p4 = self.fpt_p4(fpt_p4_out)
        #fpt_p3 = self.fpt_p3(fpt_p3_out)
        #fpt_p2 = self.fpt_p2(fpt_p2_out)
        #F.relu()
        '''
        fpt_p5 = drop_block(self.fpt_p5(fpt_p5_out))
        fpt_p4 = drop_block(self.fpt_p4(fpt_p4_out))
        fpt_p3 = drop_block(self.fpt_p3(fpt_p3_out))
        fpt_p2 = drop_block(self.fpt_p2(fpt_p2_out))
        '''
        fpt_features = [fpt_p3, fpt_p4, fpt_p5]
        
        
        return fpt_features  # 输出：经过fpt之后的对应2-5特征图
      
class LastLevelMaxPool(nn.Module):
    def forward(self, x):
        return [F.max_pool2d(x, 1, 2, 0)]


class LastLevelP6P7(nn.Module):
    """
    This module is used in RetinaNet to generate extra layers, P6 and P7.
    """
    def __init__(self, in_channels, out_channels):
        super(LastLevelP6P7, self).__init__()
        self.p6 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        for module in [self.p6, self.p7]:
            nn.init.kaiming_uniform_(module.weight, a=1)
            nn.init.constant_(module.bias, 0)
        self.use_P5 = in_channels == out_channels

    def forward(self, p5):
        x = p5 
        p6 = self.p6(x)
        p7 = self.p7(F.relu(p6))
        return [p6, p7]