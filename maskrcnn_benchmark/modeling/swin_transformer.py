import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
import pdb
# for using, the two-layer feature maps need to be concatenated together

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    #print (x.shape)
    #pdb.set_trace()
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """

    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, position_emd=False, attn_drop=0., qk_scale=None, proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale =  head_dim ** -0.5 
        self.position_emd = position_emd 
        
        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        
        #self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        #self.kv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        #self.proj = nn.Linear(dim, dim)
        #self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None,dropout=False):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = q.shape

        #qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        #q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        q = q.reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        
        attn = (q @ k.transpose(-2, -1))
        
        if self.position_emd == True:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)

        
        if mask is not None:
            nW = mask.shape[0]
            #pdb.set_trace()  
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        
        if dropout == True:
          attn = self.attn_drop(attn)
        #pdb.set_trace()
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        #x = self.proj(x)
        #x = self.proj_drop(x)
        return x


    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops



class Swin_transformerBlock(nn.Module):
    def __init__(self, out_channels, window_size, shift_size, num_heads=1,position_emd=False):
        super(Swin_transformerBlock, self).__init__()
        self.out_channels = out_channels
        
        self.num_heads = num_heads

        assert self.out_channels % self.num_heads == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

        #self.rel_h = nn.Parameter(torch.randn(out_channels // 2, 1, 1, kernel_size, 1), requires_grad=True)
        #self.rel_w = nn.Parameter(torch.randn(out_channels // 2, 1, 1, 1, kernel_size), requires_grad=True)

        #self.input_resolution = input_resolution
                
        self.shift_size = shift_size
        self.window_size = window_size
            
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.attn_drop = 0.2
        
        self.attn = WindowAttention(
            out_channels, window_size=to_2tuple(self.window_size), num_heads=num_heads,position_emd=position_emd, attn_drop=self.attn_drop)

    

        

        #self.register_buffer("attn_mask", attn_mask)


        

    def forward(self, q_out, k_out, v_out, C, H, W, x_low_padding_h,x_low_padding_w,dropout=False):

        # cyclic shift
        
        if self.shift_size > 0:          
            shifted_q = torch.roll(q_out, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_k = torch.roll(k_out, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_v = torch.roll(v_out, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        else:
            shifted_q = q_out
            shifted_k = q_out
            shifted_v = q_out

        # partition windows
        #print (x_high_padding_h,x_high_padding_w)
        q_windows = window_partition(shifted_q, self.window_size)  # nW*B, window_size, window_size, C
        q_windows = q_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        #print (q_windows.shape)
        k_windows = window_partition(shifted_k, self.window_size)  # nW*B, window_size, window_size, C
        k_windows = k_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        v_windows = window_partition(shifted_v, self.window_size)  # nW*B, window_size, window_size, C
        v_windows = v_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        #pdb.set_trace()
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
           
            img_mask = torch.zeros((1, H+x_low_padding_h, W+x_low_padding_w, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h_ in h_slices:
                for w_ in w_slices:
                    img_mask[:, h_, w_, :] = cnt
                    cnt += 1
                    
            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            attn_mask = attn_mask.cuda()
            #pdb.set_trace()
        else:
            attn_mask = None

        attn_windows = self.attn(q_windows, k_windows, v_windows, mask=attn_mask,dropout=dropout)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        

        shifted_x = window_reverse(attn_windows, self.window_size, H+x_low_padding_h, W+x_low_padding_w)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x


        return x





class Swin_transformer(nn.Module):
    def __init__(self, in_channels, out_channels, window_size, num_heads=1, position_emd=False, bias=False):
        super(Swin_transformer, self).__init__()
        self.out_channels = out_channels
        
        self.num_heads = num_heads

        assert self.out_channels % self.num_heads == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"
        self.window_size = window_size
       
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=bias)
        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1,padding=0, bias=bias)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=bias)
        
        self.Swin_transformerB1 = Swin_transformerBlock(out_channels, window_size, 0, num_heads=num_heads, position_emd=position_emd)
        self.Swin_transformerB2 = Swin_transformerBlock(out_channels, window_size, window_size//2, num_heads=num_heads, position_emd=position_emd)
        #self.Swin_transformerB3 = Swin_transformerBlock(out_channels, window_size, window_size//4, num_heads=num_heads)
        #self.Swin_transformerB4 = Swin_transformerBlock(out_channels, window_size, (window_size//2)+(window_size//4), num_heads=num_heads)
        self.reset_parameters()

    def forward(self, x_low, x_high):
                
        
        #stride = x_low.shape[-1]//x_high.shape[-1]
    
        B, C, H, W = x_low.size()
        x_low_padding_h = (self.window_size - H%self.window_size)%self.window_size
        x_low_padding_w = (self.window_size - W%self.window_size)%self.window_size
        q_out = self.query_conv(x_low)
        #k_out = self.key_conv(x_high)
        k_out = self.key_conv(x_high)
        v_out = self.value_conv(x_high)
        #v_out = k_out
        
        if x_low_padding_h !=0 or x_low_padding_w!=0:
            #pdb.set_trace()
            q_out = F.pad(q_out, [0, x_low_padding_w, 0, x_low_padding_h])
        
        

        #if stride > 1:
            #x_low = nn.AvgPool2d((stride,stride),stride)(x_low)
        #    k_out = F.interpolate(k_out,scale_factor=stride,mode="nearest")
        #    v_out = F.interpolate(v_out,scale_factor=stride,mode="nearest")
        

        #v_out = k_out
        _, _, H_high, W_high = k_out.size()
        x_high_padding_h = (self.window_size - H_high%self.window_size)%self.window_size
        x_high_padding_w = (self.window_size - W_high%self.window_size)%self.window_size
        if x_high_padding_h !=0 or x_high_padding_w!=0:
            k_out = F.pad(k_out, [0, x_high_padding_w, 0, x_high_padding_h])
            v_out = F.pad(v_out, [0, x_high_padding_w, 0, x_high_padding_h])
        #padded_x = F.pad(x_high, [self.padding, self.padding, self.padding, self.padding])
        #pdb.set_trace()
        q_out = q_out.permute(0,2,3,1)
        k_out = k_out.permute(0,2,3,1)
        v_out = v_out.permute(0,2,3,1)

        x1 = self.Swin_transformerB1(q_out, k_out, v_out, C, H, W, x_low_padding_h,x_low_padding_w)
        x2 = self.Swin_transformerB2(q_out, k_out, v_out, C, H, W, x_low_padding_h,x_low_padding_w)
        #x3 = self.Swin_transformerB3(q_out, k_out, v_out, C, H, W, x_low_padding_h,x_low_padding_w)
        #x4 = self.Swin_transformerB4(q_out, k_out, v_out, C, H, W, x_low_padding_h,x_low_padding_w)
        x = (x1+x2)/2 # (x1+x2+x3+x4)/4
        x = x.view(B, H+x_low_padding_h, W+x_low_padding_w, C)
        if x_low_padding_h !=0 or x_low_padding_w!=0:
            x=x[:,0:H,0:W,:]
        x = x.permute(0,3, 1,2)
        return x
        

    def reset_parameters(self):
        init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        #init.kaiming_normal_(self.value_conv.weight, mode='fan_out', nonlinearity='relu') 
        init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')

        #init.xavier_uniform_(self.key_conv.weight,gain=1.0)
        #init.xavier_uniform_(self.value_conv.weight,gain=1.0)
        #init.xavier_uniform_(self.query_conv.weight,gain=1.0)
        #init.normal_(self.rel_h, 0, 1)
        #init.normal_(self.rel_w, 0, 1)



class Swin_transformer_up(nn.Module):
    def __init__(self, in_channels, out_channels, window_size, scale,num_heads=1, position_emd=False, bias=False):
        super(Swin_transformer_up, self).__init__()
        self.out_channels = out_channels
        
        self.num_heads = num_heads
        self.scale = scale

        assert self.out_channels % self.num_heads == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"
        self.window_size = window_size
       
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=bias)
        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=bias)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=bias)
        #self.key_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2,padding=0, bias=bias)
        #self.value_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2,padding=0,bias=bias)
        
        self.Swin_transformerB1 = Swin_transformerBlock(out_channels, window_size, 0, num_heads=num_heads, position_emd=position_emd)
        self.Swin_transformerB2 = Swin_transformerBlock(out_channels, window_size, window_size//2, num_heads=num_heads, position_emd=position_emd)
        


        self.reset_parameters()

    def forward(self,x_low,x_high):
        
        
        B, C, H, W = x_low.size()
        x_low_padding_h = (self.window_size - H%self.window_size)%self.window_size
        x_low_padding_w = (self.window_size - W%self.window_size)%self.window_size
        q_out = self.query_conv(x_low)
        x_high = F.interpolate(x_high,scale_factor=self.scale, mode="nearest")
        
        k_out = self.key_conv(x_high)
        v_out = self.value_conv(x_high)
        #v_out = k_out
        #pdb.set_trace()
        if x_low_padding_h !=0 or x_low_padding_w!=0:
            #pdb.set_trace()
            q_out = F.pad(q_out, [0, x_low_padding_w, 0, x_low_padding_h])
        
        

        #v_out = k_out
        _, _, H_high, W_high = k_out.size()
        x_high_padding_h = (self.window_size - H_high%self.window_size)%self.window_size
        x_high_padding_w = (self.window_size - W_high%self.window_size)%self.window_size
        if x_high_padding_h !=0 or x_high_padding_w!=0:
            k_out = F.pad(k_out, [0, x_high_padding_w, 0, x_high_padding_h])
            v_out = F.pad(v_out, [0, x_high_padding_w, 0, x_high_padding_h])
        #padded_x = F.pad(x_high, [self.padding, self.padding, self.padding, self.padding])
        #pdb.set_trace()
        q_out = q_out.permute(0,2,3,1)
        k_out = k_out.permute(0,2,3,1)
        v_out = v_out.permute(0,2,3,1)
    
        x1 = self.Swin_transformerB1(q_out, k_out, v_out, C, H, W, x_low_padding_h,x_low_padding_w)
        x2 = self.Swin_transformerB2(q_out, k_out, v_out, C, H, W, x_low_padding_h,x_low_padding_w)
        x = (x1+x2)/2
        x = x.view(B, H+x_low_padding_h, W+x_low_padding_w, C)
        if x_low_padding_h !=0 or x_low_padding_w!=0:
            x=x[:,0:H,0:W,:]
        x = x.permute(0,3, 1,2)
        return x
        

    def reset_parameters(self):
        init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.value_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')
        #init.xavier_uniform_(self.key_conv.weight,gain=1.0)
        #init.xavier_uniform_(self.value_conv.weight,gain=1.0)
        #init.xavier_uniform_(self.query_conv.weight,gain=1.0)
        #init.normal_(self.rel_h, 0, 1)
        #init.normal_(self.rel_w, 0, 1)
