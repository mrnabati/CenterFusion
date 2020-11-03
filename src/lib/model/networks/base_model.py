from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from ..utils import _topk, _tranpose_and_gather_feat
from utils.ddd_utils import get_pc_hm
from utils.pointcloud import generate_pc_hm

import torch
from torch import nn

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class BaseModel(nn.Module):
    def __init__(self, heads, head_convs, num_stacks, last_channel, opt=None):
        super(BaseModel, self).__init__()
        self.opt = opt
        if opt is not None and opt.head_kernel != 3:
          print('Using head kernel:', opt.head_kernel)
          head_kernel = opt.head_kernel
        else:
          head_kernel = 3
        
        self.num_stacks = num_stacks
        self.heads = heads
        self.secondary_heads = opt.secondary_heads
        
        last_channels = {head: last_channel for head in heads}
        for head in self.secondary_heads:
          last_channels[head] = last_channel+len(opt.pc_feat_lvl)
        
        for head in self.heads:
          classes = self.heads[head]
          head_conv = head_convs[head]
          if len(head_conv) > 0:
            out = nn.Conv2d(head_conv[-1], classes, 
                  kernel_size=1, stride=1, padding=0, bias=True)
            conv = nn.Conv2d(last_channels[head], head_conv[0],
                              kernel_size=head_kernel, 
                              padding=head_kernel // 2, bias=True)
            convs = [conv]
            for k in range(1, len(head_conv)):
                convs.append(nn.Conv2d(head_conv[k - 1], head_conv[k], 
                              kernel_size=1, bias=True))
            if len(convs) == 1:
              fc = nn.Sequential(conv, nn.ReLU(inplace=True), out)
            elif len(convs) == 2:
              fc = nn.Sequential(
                convs[0], nn.ReLU(inplace=True), 
                convs[1], nn.ReLU(inplace=True), out)
            elif len(convs) == 3:
              fc = nn.Sequential(
                  convs[0], nn.ReLU(inplace=True), 
                  convs[1], nn.ReLU(inplace=True), 
                  convs[2], nn.ReLU(inplace=True), out)
            elif len(convs) == 4:
              fc = nn.Sequential(
                  convs[0], nn.ReLU(inplace=True), 
                  convs[1], nn.ReLU(inplace=True), 
                  convs[2], nn.ReLU(inplace=True), 
                  convs[3], nn.ReLU(inplace=True), out)
            if 'hm' in head:
              fc[-1].bias.data.fill_(opt.prior_bias)
            else:
              fill_fc_weights(fc)
          else:
            fc = nn.Conv2d(last_channels[head], classes, 
                kernel_size=1, stride=1, padding=0, bias=True)
            if 'hm' in head:
              fc.bias.data.fill_(opt.prior_bias)
            else:
              fill_fc_weights(fc)

          self.__setattr__(head, fc)


    def img2feats(self, x):
      raise NotImplementedError
    

    def imgpre2feats(self, x, pre_img=None, pre_hm=None):
      raise NotImplementedError


    def forward(self, x, pc_hm=None, pc_dep=None, calib=None):
      ## extract features from image
      feats = self.img2feats(x)
      out = []
      
      for s in range(self.num_stacks):
        z = {}

        ## Run the first stage heads
        for head in self.heads:
          if head not in self.secondary_heads:
            z[head] = self.__getattr__(head)(feats[s])

        if self.opt.pointcloud:
          ## get pointcloud heatmap
          if not self.training:
            if self.opt.disable_frustum:
              pc_hm = pc_dep
              if self.opt.normalize_depth:
                pc_hm[self.opt.pc_feat_channels['pc_dep']] /= self.opt.max_pc_dist
            else:
              pc_hm = generate_pc_hm(z, pc_dep, calib, self.opt)
          ind = self.opt.pc_feat_channels['pc_dep']
          z['pc_hm'] = pc_hm[:,ind,:,:].unsqueeze(1)

          ## Run the second stage heads  
          sec_feats = [feats[s], pc_hm]
          sec_feats = torch.cat(sec_feats, 1)
          for head in self.secondary_heads:
            z[head] = self.__getattr__(head)(sec_feats)
        
        out.append(z)

      return out

