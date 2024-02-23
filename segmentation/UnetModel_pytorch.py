import torch.nn as nn
import torch
import torch.nn.functional as F
from torchmetrics import JaccardIndex,Dice
class Model(nn.Module):
    def __init__(self,image_size,learning_rate):
        super(Model,self).__init__()
        self.input_size = (image_size, image_size)
        self.learning_rate = learning_rate
    def forward(self, x):
        print("do the train")

    def dice_coef(self,y_true,y_pred):
        dice = Dice()
        return dice(y_true,y_pred)
    
    def jaccard_index(self, y_true, y_pred):
        jaccard = JaccardIndex()
        return jaccard(y_true,y_pred)
    
    def dice_loss(self,y_true,y_pred):
        return 1.0 - self.dice_coef
    
    def block2d(self,in_ch,out_ch):      
        double_conv = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,kernel_size = 3,padding = 0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch,out_ch,kernel_size = 3,padding = 0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        return double_conv

    def residual_block(self,in_ch,out_ch):
        residual_conv  = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,kernel_size = 3,padding = 0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch,out_ch,kernel_size = 3,padding = 0),
            nn.BatchNorm2d(out_ch),
        )

        return residual_conv

    def upsampling_block(self, in_ch,out_ch):
        up_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_ch,out_ch,kernel_size = 3,padding = 0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        return up_conv


