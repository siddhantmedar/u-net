#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
from torchvision.transforms import CenterCrop


class Encoder(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0
            ),
            nn.ReLU(),
        )
        self.center_crop1 = CenterCrop(size=392)

        self.block2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0
            ),
            nn.ReLU(),
        )
        self.center_crop2 = CenterCrop(size=200)

        self.block3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0
            ),
            nn.ReLU(),
        )
        self.center_crop3 = CenterCrop(size=104)

        self.block4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=0
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=0
            ),
            nn.ReLU(),
        )
        self.center_crop4 = CenterCrop(size=56)

        self.block5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=0
            ),
            nn.ReLU(),
        )

    def forward(self, x, enc_features):
        enc_features.clear()

        out1 = self.block1(x)
        enc_features.append(self.center_crop1.forward(out1))
        out2 = self.block2(out1)
        enc_features.append(self.center_crop2.forward(out2))
        out3 = self.block3(out2)
        enc_features.append(self.center_crop3.forward(out3))
        out4 = self.block4(out3)
        enc_features.append(self.center_crop4.forward(out4))
        out5 = self.block5(out4)

        return out5


class Latent(nn.Module):
    def __init__(self):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=0
            ),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.block(x)
        return out


class Decoder(nn.Module):
    def __init__(self, out_channels):
        super().__init__()

        self.upconv1 = nn.ConvTranspose2d(
            in_channels=1024, out_channels=512, kernel_size=2, stride=2, padding=0
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=0
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=0
            ),
            nn.ReLU(),
        )

        self.upconv2 = nn.ConvTranspose2d(
            in_channels=512, out_channels=256, kernel_size=2, stride=2, padding=0
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=0
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0
            ),
            nn.ReLU(),
        )

        self.upconv3 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128, kernel_size=2, stride=2, padding=0
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=0
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0
            ),
            nn.ReLU(),
        )

        self.upconv4 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=2, stride=2, padding=0
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=0
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0
            ),
            nn.ReLU(),
        )

        self.conv5 = nn.Conv2d(
            in_channels=64,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x, enc_features):
        out1 = self.upconv1(x)
        out1 = torch.cat([out1, enc_features[-1]], dim=1)
        out1 = self.conv1(out1)

        out2 = self.upconv2(out1)
        out2 = torch.cat([out2, enc_features[-2]], dim=1)
        out2 = self.conv2(out2)

        out3 = self.upconv3(out2)
        out3 = torch.cat([out3, enc_features[-3]], dim=1)
        out3 = self.conv3(out3)

        out4 = self.upconv4(out3)
        out4 = torch.cat([out4, enc_features[-4]], dim=1)
        out4 = self.conv4(out4)

        out5 = self.conv5(out4)

        return out5


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=2):
        super().__init__()
        self.encoder = Encoder(in_channels)
        self.latent = Latent()
        self.decoder = Decoder(out_channels)

        self.encoder_features = list()

    def forward(self, x):
        enc_out = self.encoder(x, self.encoder_features)

        latent_out = self.latent(enc_out)

        dec_out = self.decoder(latent_out, self.encoder_features)

        return dec_out
