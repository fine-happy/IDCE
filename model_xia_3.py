import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from attention import cbam_block, eca_block, se_block, CA_Block
attention_block = [se_block, cbam_block, eca_block, CA_Block]


#-------------------------------------------------#
#   卷积块 -> 卷积 + 标准化 + 激活函数
#   Conv2d + BatchNormalization + LeakyReLU
#-------------------------------------------------#
class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BasicConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


#---------------------------------------------------#
#   卷积 + 上采样
#---------------------------------------------------#
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            BasicConv(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x,):
        x = self.upsample(x)
        return x




class CSDN_Tem(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(CSDN_Tem, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out
		
		
class CSDN_Tem_xia(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(CSDN_Tem_xia, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=2,
            stride=2,
            padding=0,
            groups=1
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out

class enhance_net_nopool(nn.Module):

	def __init__(self,scale_factor,phi=3):
		super(enhance_net_nopool, self).__init__()
		self.phi = phi

		self.relu = nn.ReLU(inplace=True)
		self.scale_factor = scale_factor
		# self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)
		number_f = 32

#   zerodce DWC + p-shared
# 		self.e_conv1 = CSDN_Tem(3,number_f)
# 		self.e_conv2 = CSDN_Tem(number_f,number_f)
# 		self.e_conv3 = CSDN_Tem(number_f,number_f)
# 		self.e_conv4 = CSDN_Tem(number_f,number_f)
# 		self.e_conv5 = CSDN_Tem(number_f*2,number_f)
# 		self.e_conv6 = CSDN_Tem(number_f*2,number_f)
# 		self.e_conv7 = CSDN_Tem(number_f*2,3)



		self.e_conv1 =  CSDN_Tem(3, number_f)
		#  512,512,3 -> 512,512,32
		self.e_conv2 = CSDN_Tem(number_f, number_f)
		#  512,512,32 -> 512,512,32
		self.e_conv3 = CSDN_Tem(number_f*2, number_f)
		#  512,512,64 -> 512,512,32

		self.e_conv3_xia = CSDN_Tem_xia(number_f,number_f*2)
		#  512,512,32 -> 256,256,64
		self.e_conv4 = CSDN_Tem(number_f*2,number_f*2)
		#  256,256,64 -> 256,256,64
		self.e_conv5 = CSDN_Tem(number_f*4,number_f*2)
		#  256,256,128 -> 256,256,64

		self.e_conv6_up = Upsample(number_f*2, number_f)
		# 256,256,64 -> 512,512,32

		self.e_conv7 = CSDN_Tem(number_f,3)
		# 512,512,32  -> 512,512,3
        #

		if 1 <= self.phi and self.phi <= 4:
			self.feat1_upsample_att = attention_block[self.phi - 1](32)
			self.feat2_upsample_att = attention_block[self.phi - 1](64)



	def enhance(self, x,x_r):

		x_1 = x + x_r*(torch.pow(x,2)-x)
		x_2 = x_1 + x_r*(torch.pow(x_1,2)-x_1)
		x_3 = x_2 + x_r*(torch.pow(x_2,2)-x_2)
		enhance_image_1 = x_3 + x_r*(torch.pow(x_3,2)-x_3)
		x_4 = enhance_image_1 + x_r*(torch.pow(enhance_image_1,2)-enhance_image_1)
		x_5 = x_4 + x_r*(torch.pow(x_4,2)-x_4)
		x_6 = x_5 + x_r*(torch.pow(x_5,2)-x_5)
		enhance_image = x_6 + x_r*(torch.pow(x_6,2)-x_6)

		return enhance_image , enhance_image_1 , x_1 , x_2 , x_3 , x_4 , x_5 , x_6
		# return enhance_image

	def forward(self, x):
		if self.scale_factor==1:
			x_down = x
		else:
			x_down = F.interpolate(x,scale_factor=1/self.scale_factor, mode='bilinear')



		# x1 = self.relu(self.e_conv1(x_down))
		# x2 = self.relu(self.e_conv2(x1))
		# x3 = self.relu(self.e_conv3(x2))
		# x4 = self.relu(self.e_conv4(x3))
		# x5 = self.relu(self.e_conv5(torch.cat([x3,x4],1)))
		# x6 = self.relu(self.e_conv6(torch.cat([x2,x5],1)))
		# x_r = F.tanh(self.e_conv7(torch.cat([x1,x6],1)))

		x1 = self.relu(self.e_conv1(x_down))
		#  512,512,3 -> 512,512,32
		x2 = self.relu(self.e_conv2(x1))
		#  512,512,32 -> 512,512,32
		x3 = self.relu(self.e_conv3(torch.cat([x1, x2], 1)))
		#  512,512,32 -> 512,512,64 -> 512,512,32

		x4 = self.relu(self.e_conv3_xia(x3))
		#  512,512,32 -> 256,256,64

		if 1 <= self.phi and self.phi <= 4:
			x4 = self.feat2_upsample_att(x4)


		x5 = self.relu(self.e_conv4(x4))
		#  256,256,64 -> 256,256,64

		x6 = self.relu(self.e_conv5(torch.cat([x4, x5], 1)))
		#  256,256,64 -> 256,256,128 -> 256,256,64

		Upsample_1 = self.relu(self.e_conv6_up(x6))
		#   256,256,64 -> 512,512,32
		# if 1 <= self.phi and self.phi <= 4:
		#  	Upsample_1 = self.feat1_upsample_att(Upsample_1)
		#  512,512,32 -> 512,512,32
		x7 = self.relu(self.e_conv3(torch.cat([x3, Upsample_1], 1)))
		#  512,512,32 -> 512,512,64 -> 512,512,32

		x_r = F.tanh(self.e_conv7(x7))
		#  512,512,32 -> 512,512,3
		
		# if self.scale_factor==1:
		# 	x_r = x_r
		# else:
		# 	x_r = self.upsample(x_r)
		enhance_image , enhance_image_1 , x_1 , x_2 , x_3 , x_4 , x_5 , x_6 = self.enhance(x,x_r)
		# enhance_image = self.enhance(x,x_r)
		return enhance_image , enhance_image_1 , x_1 , x_2 , x_3 , x_4 , x_5 , x_6 , x_r
