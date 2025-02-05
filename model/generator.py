import torch
from torch import nn
from torch.nn import functional as F


class EncoderBlock(nn.Module):
    """Encoder block"""

    def __init__(self, inplanes, outplanes, kernel_size=4, stride=2, padding=1, norm=True):
        """
        Initializes the generator layer with the given parameters.
        Args:
            inplanes (int): Number of input channels.
            outplanes (int): Number of output channels.
            kernel_size (int, optional): Size of the convolving kernel. Default is 4.
            stride (int, optional): Stride of the convolution. Default is 2.
            padding (int, optional): Zero-padding added to both sides of the input. Default is 1.
            norm (bool, optional): If True, applies batch normalization. Default is True.
        """
        super().__init__()
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.conv = nn.Conv2d(inplanes, outplanes, kernel_size, stride, padding)
        
        self.bn=None
        if norm:
            self.bn = nn.BatchNorm2d(outplanes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the generator network.
        Args:
            x (torch.Tensor): Input tensor to the generator network.
        Returns:
            torch.Tensor: Output tensor after applying leaky ReLU, convolution, 
                          and optional batch normalization.
        """
        fx = self.lrelu(x)
        fx = self.conv(fx)
        
        if self.bn is not None:
            fx = self.bn(fx)
            
        return fx

    
class DecoderBlock(nn.Module):
    """Decoder block"""

    def __init__(self, inplanes, outplanes, kernel_size=4, stride=2, padding=1, dropout=False):
        """
        Initializes the generator layer with the specified parameters.
        Args:
            inplanes (int): Number of input channels.
            outplanes (int): Number of output channels.
            kernel_size (int, optional): Size of the convolving kernel. Default is 4.
            stride (int, optional): Stride of the convolution. Default is 2.
            padding (int, optional): Zero-padding added to both sides of the input. Default is 1.
            dropout (bool, optional): If True, applies dropout with a probability of 0.5. Default is False.
        """
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.deconv = nn.ConvTranspose2d(inplanes, outplanes, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(outplanes)       
        
        self.dropout=None
        if dropout:
            self.dropout = nn.Dropout2d(p=0.5, inplace=True)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the generator model.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor after applying ReLU activation, 
                          deconvolution, batch normalization, and optional dropout.
        """
        fx = self.relu(x)
        fx = self.deconv(fx)
        fx = self.bn(fx)

        if self.dropout is not None:
            fx = self.dropout(fx)
            
        return fx

    
class Generator(nn.Module):
    """Encoder-Decoder model"""

    def __init__(self):
        """
        Initializes the generator model for the Pix2Pix architecture.
        The generator consists of an encoder-decoder structure with convolutional layers.
        The encoder compresses the input image into a lower-dimensional representation,
        while the decoder reconstructs the image from this representation.
        Encoder layers:
            encoder1 (nn.Conv2d): First convolutional layer with 3 input channels and 64 output channels.
            encoder2 (EncoderBlock): Encoder block with 64 input channels and 128 output channels.
            encoder3 (EncoderBlock): Encoder block with 128 input channels and 256 output channels.
            encoder4 (EncoderBlock): Encoder block with 256 input channels and 512 output channels.
            encoder5 (EncoderBlock): Encoder block with 512 input channels and 512 output channels.
            encoder6 (EncoderBlock): Encoder block with 512 input channels and 512 output channels.
            encoder7 (EncoderBlock): Encoder block with 512 input channels and 512 output channels.
            encoder8 (EncoderBlock): Encoder block with 512 input channels and 512 output channels, without normalization.

        Decoder layers:
            decoder8 (DecoderBlock): Decoder block with 512 input channels and 512 output channels, with dropout.
            decoder7 (DecoderBlock): Decoder block with 512 input channels and 512 output channels, with dropout.
            decoder6 (DecoderBlock): Decoder block with 512 input channels and 512 output channels, with dropout.
            decoder5 (DecoderBlock): Decoder block with 512 input channels and 512 output channels.
            decoder4 (DecoderBlock): Decoder block with 512 input channels and 256 output channels.
            decoder3 (DecoderBlock): Decoder block with 256 input channels and 128 output channels.
            decoder2 (DecoderBlock): Decoder block with 128 input channels and 64 output channels.
            decoder1 (nn.ConvTranspose2d): Final transposed convolutional layer with 64 input channels and 3 output channels.
        """
        super().__init__()
        
        self.encoder1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.encoder2 = EncoderBlock(64, 128)
        self.encoder3 = EncoderBlock(128, 256)
        self.encoder4 = EncoderBlock(256, 512)
        self.encoder5 = EncoderBlock(512, 512)
        self.encoder6 = EncoderBlock(512, 512)
        self.encoder7 = EncoderBlock(512, 512)
        self.encoder8 = EncoderBlock(512, 512, norm=False)
        
        self.decoder8 = DecoderBlock(512, 512, dropout=True)
        self.decoder7 = DecoderBlock(512, 512, dropout=True)
        self.decoder6 = DecoderBlock(512, 512, dropout=True)
        self.decoder5 = DecoderBlock(512, 512)
        self.decoder4 = DecoderBlock(512, 256)
        self.decoder3 = DecoderBlock(256, 128)
        self.decoder2 = DecoderBlock(128, 64)
        self.decoder1 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the generator network.
        Args:
            x (torch.Tensor): Input tensor to the generator network.
        Returns:
            torch.Tensor: Output tensor after passing through the generator network.
        """
        # encoder forward
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        e6 = self.encoder6(e5)
        e7 = self.encoder7(e6)
        e8 = self.encoder8(e7)
        # decoder forward
        d8 = self.decoder8(e8)
        d7 = self.decoder7(d8)
        d6 = self.decoder6(d7)
        d5 = self.decoder5(d6)
        d4 = self.decoder4(d5)
        d3 = self.decoder3(d4)
        d2 = F.relu(self.decoder2(d3))
        d1 = self.decoder1(d2)
        
        return torch.tanh(d1)
    
    
class UnetGenerator(nn.Module):
    """Unet-like Encoder-Decoder model"""

    def __init__(self):
        """
        Initializes the generator model for the Pix2Pix architecture.
        The generator consists of an encoder-decoder structure with skip connections.
        The encoder uses a series of convolutional layers to downsample the input image,
        while the decoder uses transposed convolutional layers to upsample the feature maps
        back to the original image size.
        Encoder layers:
            - encoder1: Convolutional layer with 3 input channels and 64 output channels.
            - encoder2: EncoderBlock with 64 input channels and 128 output channels.
            - encoder3: EncoderBlock with 128 input channels and 256 output channels.
            - encoder4: EncoderBlock with 256 input channels and 512 output channels.
            - encoder5: EncoderBlock with 512 input channels and 512 output channels.
            - encoder6: EncoderBlock with 512 input channels and 512 output channels.
            - encoder7: EncoderBlock with 512 input channels and 512 output channels.
            - encoder8: EncoderBlock with 512 input channels and 512 output channels, no normalization.
        Decoder layers:
            - decoder8: DecoderBlock with 512 input channels and 512 output channels, with dropout.
            - decoder7: DecoderBlock with 1024 input channels and 512 output channels, with dropout.
            - decoder6: DecoderBlock with 1024 input channels and 512 output channels, with dropout.
            - decoder5: DecoderBlock with 1024 input channels and 512 output channels.
            - decoder4: DecoderBlock with 1024 input channels and 256 output channels.
            - decoder3: DecoderBlock with 512 input channels and 128 output channels.
            - decoder2: DecoderBlock with 256 input channels and 64 output channels.
            - decoder1: Transposed convolutional layer with 128 input channels and 3 output channels.
        """
        super().__init__()
        
        self.encoder1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.encoder2 = EncoderBlock(64, 128)
        self.encoder3 = EncoderBlock(128, 256)
        self.encoder4 = EncoderBlock(256, 512)
        self.encoder5 = EncoderBlock(512, 512)
        self.encoder6 = EncoderBlock(512, 512)
        self.encoder7 = EncoderBlock(512, 512)
        self.encoder8 = EncoderBlock(512, 512, norm=False)
        
        self.decoder8 = DecoderBlock(512, 512, dropout=True)
        self.decoder7 = DecoderBlock(2*512, 512, dropout=True)
        self.decoder6 = DecoderBlock(2*512, 512, dropout=True)
        self.decoder5 = DecoderBlock(2*512, 512)
        self.decoder4 = DecoderBlock(2*512, 256)
        self.decoder3 = DecoderBlock(2*256, 128)
        self.decoder2 = DecoderBlock(2*128, 64)
        self.decoder1 = nn.ConvTranspose2d(2*64, 3, kernel_size=4, stride=2, padding=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the generator network.
        Args:
            x (torch.Tensor): Input tensor to the generator network.
        Returns:
            torch.Tensor: Output tensor after passing through the generator network.
        """
        # encoder forward
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        e6 = self.encoder6(e5)
        e7 = self.encoder7(e6)
        e8 = self.encoder8(e7)
        # decoder forward + skip connections
        d8 = self.decoder8(e8)
        d8 = torch.cat([d8, e7], dim=1)
        d7 = self.decoder7(d8)
        d7 = torch.cat([d7, e6], dim=1)
        d6 = self.decoder6(d7)
        d6 = torch.cat([d6, e5], dim=1)
        d5 = self.decoder5(d6)
        d5 = torch.cat([d5, e4], dim=1)
        d4 = self.decoder4(d5)
        d4 = torch.cat([d4, e3], dim=1)
        d3 = self.decoder3(d4)
        d3 = torch.cat([d3, e2], dim=1)
        d2 = F.relu(self.decoder2(d3))
        d2 = torch.cat([d2, e1], dim=1)
        d1 = self.decoder1(d2)
        
        return torch.tanh(d1)
    

if __name__ == '__main__':
    # test both generator models using a random input tensor
    device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(1, 3, 256, 256).to(device)

    model = Generator().to(device)
    y = model(x)
    print(y.shape)
    
    model = UnetGenerator().to(device)
    y = model(x)
    print(y.shape)
    