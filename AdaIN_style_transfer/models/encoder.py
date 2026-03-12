# -*- coding: utf-8 -*-
import torch.nn as nn
import torchvision.models as models

class VGGEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights = models.VGG19_Weights.DEFAULT).features
        
        # 按层切片，方便提取不同层的特征
        self.slice1 = vgg[:2]   # reLU1_1
        self.slice2 = vgg[2:7]  # reLU2_1   
        self.slice3 = vgg[7:12] # reLU3_1
        self.slice4 = vgg[12:21]# reLU4_1 
        
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        relu1_1 = self.slice1(x)
        relu2_1 = self.slice2(relu1_1)
        relu3_1 = self.slice3(relu2_1)
        relu4_1 = self.slice4(relu3_1)
        return [relu1_1, relu2_1, relu3_1, relu4_1]
        
