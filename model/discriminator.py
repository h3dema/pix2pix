# -*- coding: utf-8 -*-
"""

@@@  @@@  @@@@@@   @@@@@@@   @@@@@@@@  @@@@@@@@@@    @@@@@@
@@@  @@@  @@@@@@@  @@@@@@@@  @@@@@@@@  @@@@@@@@@@@  @@@@@@@@
@@!  @@@      @@@  @@!  @@@  @@!       @@! @@! @@!  @@!  @@@
!@!  @!@      @!@  !@!  @!@  !@!       !@! !@! !@!  !@!  @!@
@!@!@!@!  @!@!!@   @!@  !@!  @!!!:!    @!! !!@ @!@  @!@!@!@!
!!!@!!!!  !!@!@!   !@!  !!!  !!!!!:    !@!   ! !@!  !!!@!!!!
!!:  !!!      !!:  !!:  !!!  !!:       !!:     !!:  !!:  !!!
:!:  !:!      :!:  :!:  !:!  :!:       :!:     :!:  :!:  !:!
::   :::  :: ::::   :::: ::   :: ::::  :::     ::   ::   :::
 :   : :   : : :   :: :  :   : :: ::    :      :     :   : :


Copyright 2025 by Henrique Duarte Moura.
All rights reserved.

This file is part of pix2pix,
and is released under the "MIT License Agreement".
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

Please see the LICENSE file that should have been included as part of this package.
"""
__maintainer__ = "Henrique Duarte Moura"
__email__ = "h3dema@outlook.com"
__version__ = "0.1.0"

import torch
from torch import nn
from torch.nn import functional as F

        
class BasicBlock(nn.Module):
    """Basic block"""

    def __init__(self, inplanes, outplanes, kernel_size=4, stride=2, padding=1, norm=True):
        """
        Initializes the Discriminator layer with a convolutional layer, optional instance normalization, and a LeakyReLU activation.

        Args:
            inplanes (int): Number of input channels.
            outplanes (int): Number of output channels.
            kernel_size (int, optional): Size of the convolving kernel. Default is 4.
            stride (int, optional): Stride of the convolution. Default is 2.
            padding (int, optional): Zero-padding added to both sides of the input. Default is 1.
            norm (bool, optional): If True, applies instance normalization. Default is True.
        """
        super().__init__()
        self.conv = nn.Conv2d(inplanes, outplanes, kernel_size, stride, padding)
        self.isn = None
        if norm:
            self.isn = nn.InstanceNorm2d(outplanes)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the discriminator model.
        Args:
            x (torch.Tensor): Input tensor to the discriminator.
        Returns:
            torch.Tensor: Output tensor after applying convolution, 
                          instance normalization (if applicable), 
                          and leaky ReLU activation.
        """
        fx = self.conv(x)
        
        if self.isn is not None:
            fx = self.isn(fx)
            
        fx = self.lrelu(fx)
        return fx
    
    
class Discriminator(nn.Module):
    """Basic Discriminator"""

    def __init__(self):
        """
        Initializes the Discriminator model.

        This model consists of several convolutional blocks followed by a final
        convolutional layer. The blocks are defined as follows:
        
        - block1: BasicBlock with input channels 3 and output channels 64, without normalization.
        - block2: BasicBlock with input channels 64 and output channels 128.
        - block3: BasicBlock with input channels 128 and output channels 256.
        - block4: BasicBlock with input channels 256 and output channels 512.
        - block5: A final convolutional layer with input channels 512 and output channel 1,
                  kernel size of 4, stride of 1, and padding of 1.
        """
        super().__init__()
        self.block1 = BasicBlock(3, 64, norm=False)
        self.block2 = BasicBlock(64, 128)
        self.block3 = BasicBlock(128, 256)
        self.block4 = BasicBlock(256, 512)
        self.block5 = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the discriminator network.
        Args:
            x (torch.Tensor): Input tensor to the discriminator.
        Returns:
            torch.Tensor: Output tensor after passing through all the blocks.
        """
        # blocks forward
        fx = self.block1(x)
        fx = self.block2(fx)
        fx = self.block3(fx)
        fx = self.block4(fx)
        fx = self.block5(fx)
        
        return fx
    
    
class ConditionalDiscriminator(nn.Module):
    """Conditional Discriminator"""

    def __init__(self):
        """
        Initializes the Discriminator model with a series of convolutional blocks.

        The model consists of five blocks:
        - block1: A BasicBlock with input channels 6 and output channels 64, without normalization.
        - block2: A BasicBlock with input channels 64 and output channels 128.
        - block3: A BasicBlock with input channels 128 and output channels 256.
        - block4: A BasicBlock with input channels 256 and output channels 512.
        - block5: A final convolutional layer with input channels 512 and output channel 1, 
                  with a kernel size of 4, stride of 1, and padding of 1.
        """
        super().__init__()
        self.block1 = BasicBlock(6, 64, norm=False)
        self.block2 = BasicBlock(64, 128)
        self.block3 = BasicBlock(128, 256)
        self.block4 = BasicBlock(256, 512)
        self.block5 = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the discriminator model.
        Args:
            x (torch.Tensor): Input tensor representing the image.
            cond (torch.Tensor): Condition tensor to concatenate with the input tensor.
        Returns:
            torch.Tensor: Output tensor after passing through the discriminator blocks.
        """
        x = torch.cat([x, cond], dim=1)
        # blocks forward
        fx = self.block1(x)
        fx = self.block2(fx)
        fx = self.block3(fx)
        fx = self.block4(fx)
        fx = self.block5(fx)
        
        return fx
    

if __name__ == '__main__':
    # test both discriminator models using a random input tensor
    device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(1, 3, 256, 256).to(device)

    model = Discriminator().to(device)
    y = model(x)
    print(y.shape)
    
    model = ConditionalDiscriminator().to(device)
    cond = torch.randn(1, 3, 256, 256).to(device)
    y = model(x, cond)
    print(y.shape)
    