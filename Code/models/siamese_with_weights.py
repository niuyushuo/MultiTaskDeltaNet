import numpy as np # linear algebra

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import init
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms

from models import unet


class Siamese(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(Siamese, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        print(self.device)

        self.unet = unet.UNet(in_channels=3, out_channels=2, init_features=32)
        self.unet.to(self.device)

        self.checkpoint = torch.load('/home/yun13001/code/Carbon/model_reg/tianyu_new_data/code_github/cv_github/checkpoints/unet/a2_results/best_a2.pt', map_location=torch.device(self.device),weights_only=False)   #### init2 
        #self.checkpoint = torch.load('/home/yun13001/code/Carbon/model_reg/tianyu_new_data/code_github/cv_github/checkpoints/unet/a1_results/best_a1.pt', map_location=torch.device(self.device),weights_only=False)   #### init1 

        self.unet.load_state_dict(self.checkpoint['model_G_state_dict'])

        self.enc1_1 = self.unet.encoder1
        self.pool1_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc1_2 = self.unet.encoder2
        self.pool1_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc1_3 = self.unet.encoder3
        self.pool1_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc1_4 = self.unet.encoder4
        self.pool1_4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck_1 = self.unet.bottleneck

        self.enc2_1 = self.unet.encoder1
        self.pool2_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc2_2 = self.unet.encoder2
        self.pool2_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc2_3 = self.unet.encoder3
        self.pool2_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc2_4 = self.unet.encoder4
        self.pool2_4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck_2 = self.unet.bottleneck


        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
            nn.Conv2d(3, out_channels, 1),
            )

        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
            nn.Conv2d(3, out_channels, 1),
            )



    def encoder1(self,x):
        enc1 = self.enc1_1(x)
        enc2 = self.enc1_2(self.pool1_1(enc1))
        enc3 = self.enc1_3(self.pool1_2(enc2))
        enc4 = self.enc1_4(self.pool1_3(enc3))
        bottleneck = self.bottleneck_1(self.pool1_4(enc4))
        return enc1,enc2,enc3,enc4,bottleneck

    def encoder2(self,x):
        enc1 = self.enc2_1(x)
        enc2 = self.enc2_2(self.pool2_1(enc1))
        enc3 = self.enc2_3(self.pool2_2(enc2))
        enc4 = self.enc2_4(self.pool2_3(enc3))
        bottleneck = self.bottleneck_2(self.pool2_4(enc4))
        return enc1,enc2,enc3,enc4,bottleneck

    def decoder(self,enc1,enc2,enc3,enc4,bottleneck):

        dec4 = self.unet.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.unet.decoder4(dec4)
        dec3 = self.unet.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.unet.decoder3(dec3)
        dec2 = self.unet.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.unet.decoder2(dec2)
        dec1 = self.unet.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.unet.decoder1(dec1)
        #return torch.sigmoid(self.unet.conv(dec1))
        return dec1

    def forward(self,x1,x2):

        x1_enc1,x1_enc2,x1_enc3,x1_enc4,x1_bottle = self.encoder1(x1)
        x2_enc1,x2_enc2,x2_enc3,x2_enc4,x2_bottle = self.encoder2(x2)

        out1 = self.decoder(x1_enc1,x1_enc2,x1_enc3,x1_enc4,x1_bottle)
        out2 = self.decoder(x2_enc1,x2_enc2,x2_enc3,x2_enc4,x2_bottle)

        con = torch.cat((out1, out2), dim=1)
        out1 = self.conv_layer1(con)
        out2 = self.conv_layer2(con)

        return out1, out2

        return out

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )
