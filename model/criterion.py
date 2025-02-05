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


class GeneratorLoss(nn.Module):
    """Loss function for the generator model."""

    def __init__(self, alpha=100):
        """
        Initializes the criterion with specified alpha value.

        Args:
            alpha (float, optional): Weighting factor for the loss components. Default is 100.

        Attributes:
            alpha (float): Weighting factor for the loss components.
            bce (nn.BCEWithLogitsLoss): Binary Cross Entropy loss with logits.
            l1 (nn.L1Loss): L1 loss (mean absolute error).
        """
        super().__init__()
        self.alpha=alpha
        self.bce=nn.BCEWithLogitsLoss()
        self.l1=nn.L1Loss()
        
    def forward(self, fake, real, fake_pred):
        """
        Computes the loss for the given fake and real images and their corresponding predictions.

        Args:
            fake (torch.Tensor): The generated fake images.
            real (torch.Tensor): The real images.
            fake_pred (torch.Tensor): The discriminator's predictions for the fake images.

        Returns:
            torch.Tensor: The computed loss value.
        """
        fake_target = torch.ones_like(fake_pred)
        loss = self.bce(fake_pred, fake_target) + self.alpha* self.l1(fake, real)
        return loss
    
    
class DiscriminatorLoss(nn.Module):
    """Loss function for the discriminator model."""

    def __init__(self):
        """
        Initializes the criterion class.

        This constructor sets up the loss function to be Binary Cross Entropy with Logits Loss (BCEWithLogitsLoss).
        """
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()
        
    def forward(self, fake_pred, real_pred):
        """
        Computes the loss for the discriminator in a GAN.

        Args:
            fake_pred (torch.Tensor): The discriminator's predictions on the generated (fake) data.
            real_pred (torch.Tensor): The discriminator's predictions on the real data.

        Returns:
            torch.Tensor: The average loss computed from the fake and real predictions.
        """
        fake_target = torch.zeros_like(fake_pred)
        real_target = torch.ones_like(real_pred)
        fake_loss = self.loss_fn(fake_pred, fake_target)
        real_loss = self.loss_fn(real_pred, real_target)
        loss = (fake_loss + real_loss)/2
        return loss
