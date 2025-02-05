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

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import v2


class CityscapesDataset(Dataset):

    # v2.toTensor() is deprecated thus we need to define our own basic transform
    basic_transform = v2.Compose([
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True),
    ])


    def __init__(self, root_dir, mode='train', transform=None):
        """
        Args:
            root_dir (str): Root directory of the Cityscapes dataset.
            mode (str): 'train' or 'test' to select the dataset split.
            transform (callable, optional): Optional transform to be applied on an image. If provided, make sure that you added the toTensor() transform.
        """
        if mode not in ['train', 'test']:
            raise ValueError("split must be either 'train' or 'test'")
        
        self.root_dir = root_dir
        self.mode = mode
        # if no transform is provided, at least convert the image to a tensor
        self.transform = self.basic_transform if transform is None else transform
        self.image_dir = os.path.join(root_dir, mode)
        self.images = []

        self.images = glob.glob(os.path.join(self.image_dir, "*.jpg"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves the image at the specified index, splits it into two halves, 
        applies any specified transformations, and returns the two halves separately.
        Args:
            idx (int): Index of the image to retrieve.
        Returns:
            tuple: A tuple containing two halves of the image (imgA, imgB).
        """
        if idx < 0 or idx >= len(self.images):
            raise IndexError(f"Index `idx` out of range. Must be between 0 and {len(self.images) - 1}")

        img_path = self.images[idx]
        # the image is divided into two halves. We will return the two halves separately
        image = np.array(Image.open(img_path).convert("RGB"))
        imgA = image[:, :256, :]
        imgB = image[:, 256:, :]

        if self.transform:
            imgA, imgB = self.transform(imgA, imgB)
        
        return imgA, imgB  # Return both images


# Example usage:
if __name__ == "__main__":
    # basic transformation
    transform = v2.Compose([
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True),
    ])
    # create dataset
    dataset = CityscapesDataset(root_dir="datasets/cityscapes", mode='train', transform=transform)
    # retrieve first image for testing
    imgA, imgB = dataset[0]
    print(f"Image1 tensor shape: {imgA.shape}")
    print(f"Image2 tensor shape: {imgB.shape}")
