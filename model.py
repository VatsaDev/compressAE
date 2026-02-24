# basic AE, image in/out, compression

import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.ao.quantization import QuantStub, DeQuantStub, get_default_qat_qconfig
        
class Auto_Encoder(nn.Module):

    def __init__(self, input_size, latent_size, mode):
        super(Auto_Encoder, self).__init__()
        
        nc = 256
        
        self.mode = mode
        self.input_size = input_size
        self.latent_size = latent_size

        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        # Encoder
        self.enc = nn.Sequential(
            nn.Conv2d(3, nc, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(1,nc),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d((latent_size, latent_size)),
            nn.Conv2d(nc, nc//4, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(1,nc//4),
            nn.SiLU(inplace=True),
            nn.Conv2d(nc//4, nc//8, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(1,nc//8),
            nn.SiLU(inplace=True),
            nn.Conv2d(nc//8, nc//16, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(1,nc//16),
            nn.SiLU(inplace=True),
            nn.Conv2d(nc//16, 3, kernel_size=3, stride=1, padding=1),
        )

        # Decoder
        self.dec = nn.Sequential(
            nn.Conv2d(3, nc//16, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(1,nc//16),
            nn.SiLU(inplace=True),
            nn.Conv2d(nc//16, nc//8, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(1,nc//8),
            nn.SiLU(inplace=True),
            nn.Conv2d(nc//8, nc//4, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(1,nc//4),
            nn.SiLU(inplace=True),
            nn.Conv2d(nc//4, nc, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(1,nc),
            nn.SiLU(inplace=True),
            nn.Upsample(size=(input_size, input_size), mode='bilinear', align_corners=True),
            nn.Conv2d(nc, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x): # output image and latent

        if self.mode=="norm":
            encoded = self.enc(x)
            decoded = self.dec(encoded)

        if self.mode=="quant":
            x = self.quant(x)
            encoded = self.encoder(x)
            output = self.decoder(encoded)
            output = self.dequant(output)
        
        return encoded, decoded
