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

import glob
from PIL import Image
from torch.utils.data import Dataset


class MyDataset(Dataset):
    """
    A custom dataset class for loading paired images from two directories.
    Args:
        root (str): Root directory of the dataset. Default is '.'.
        transform (callable, optional): A function/transform to apply to the images.
        mode (str): Mode of the dataset, typically 'train' or 'test'. Default is 'train'.
        direction (str): Direction of the image pairing, either 'A2B' or 'B2A'. Default is 'A2B'.

    This class assumes that the images are stored with the following directory structure:

    - root/
        +-- train/
        |     +--- A/
        |     +--- B/
        |
        +-- test/
              +--- A/
              +--- B/

    Methods:
        __len__(): Returns the number of image pairs in the dataset.
        __getitem__(idx): Returns the image pair at the specified index.
    """

    def __init__(self,
                 root: str='.',
                 transform=None,
                 mode: str='train',
                 direction: str = 'A2B'):
        """
        Initialize the dataset.

        Args:
            root (str): Root directory of the dataset. Defaults to '.'.
            transform (callable, optional): A function/transform that takes in an image and returns a transformed version. Defaults to None.
            mode (str): Mode of the dataset, either 'train' or 'test'. Defaults to 'train'.
            direction (str): Direction of the transformation, either 'A2B' or 'B2A'. Defaults to 'A2B'.

        Raises:
            ValueError: If the direction is not 'A2B' or 'B2A'.
            ValueError: If the mode is not 'train' or 'test'.
            ValueError: If the number of files in directory A does not match the number of files in directory B.
        """
        self.root=root
        self.filesA=sorted(glob.glob(f"{root}/{mode}/A/*.jpg"))
        self.filesB=sorted(glob.glob(f"{root}/{mode}/B/*.jpg"))
        self.transform=transform
        self.mode=mode
        self.direction = direction
        if direction not in ['A2B', 'B2A']:
            raise ValueError(f"Invalid direction '{direction}'. Expected 'A2B' or 'B2A'.")
        if mode not in ['train', 'test']:
            raise ValueError(f"Invalid mode '{mode}'. Expected 'train' or 'test'.")
        if len(self.filesA) != len(self.filesB):
            raise ValueError("The number of files in directory A does not match the number of files in directory B.")
        
    def __len__(self,):
        return len(self.filesA)
    
    def __getitem__(self, idx):
        """
        Retrieves the paired images from the dataset at the specified index.
        Args:
            idx (int): Index of the image pair to retrieve.
        Returns:
            tuple: A tuple containing two images. If the direction is 'A2B', the tuple
                   will be (imgA, imgB). Otherwise, it will be (imgB, imgA).
        """
        imgA = Image.open(self.filesA[idx]).convert('RGB')
        imgB = Image.open(self.filesB[idx]).convert('RGB')
        W, H = imgA.size
        cW = W // 2
        
        if self.transform:
            imgA, imgB = self.transform(imgA, imgB)
            
        return (imgA, imgB) if self.direction == 'A2B' else (imgB, imgA)
