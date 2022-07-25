import torch
from torch import nn



def ConvBlock(in_channels, out_channels, stride = 2, kernel_size = 4, padding = 1, bias = True, activation = 'leakyrelu'):
    if activation == 'leakyrelu':
        activation_layer = nn.LeakyReLU(0.2)
    if activation == 'tanh':
        activation_layer = nn.Tanh()
    if activation == 'sigmoid':
        activation_layer = nn.Sigmoid()
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=stride, kernel_size=kernel_size, padding=padding, bias=bias),
        nn.BatchNorm2d(out_channels),
        activation_layer
    )

def DoubleConvBlock(in_channels, out_channels, stride = 1, kernel_size = 3, padding = 1, bias = True):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=stride, kernel_size=kernel_size, padding=padding, bias=bias),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2),
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels, stride=stride, kernel_size=kernel_size, padding=padding, bias=bias),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2)
    )

def ConvTransBlock(in_channels, out_channels, stride = 2, kernel_size = 4, padding = 1, bias = True):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, stride=stride, kernel_size=kernel_size, padding=padding, bias=bias),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2)
    )

class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        in_channels = 1
        out_channels = 3
        in_channels_down1 = in_channels
        out_channels_down1 = 32
        self.down1 = nn.Sequential(
            ConvBlock(in_channels=in_channels_down1, out_channels = out_channels_down1),
            ConvBlock(in_channels=out_channels_down1, out_channels=out_channels_down1, stride=1, kernel_size=3)
        )

        in_channels_down2 = out_channels_down1 # 32
        out_channels_down2 = in_channels_down2*2 # 64
        self.down2 = nn.Sequential(
            ConvBlock(in_channels=in_channels_down2, out_channels = out_channels_down2),
            ConvBlock(in_channels=out_channels_down2, out_channels=out_channels_down2, stride=1, kernel_size=3)
        )

        in_channels_down3 = out_channels_down2 # 64
        out_channels_down3 = in_channels_down3*2 # 128
        self.down3 = nn.Sequential(
            ConvBlock(in_channels=in_channels_down3, out_channels = out_channels_down3),
            ConvBlock(in_channels=out_channels_down3, out_channels=out_channels_down3, stride=1, kernel_size=3)
        )

        in_channels_down4 = out_channels_down3 # 128
        out_channels_down4 = in_channels_down4*2 # 256
        self.down4 = nn.Sequential(
            ConvBlock(in_channels=in_channels_down4, out_channels=out_channels_down4),
            ConvBlock(in_channels=out_channels_down4, out_channels=out_channels_down4, stride=1, kernel_size=3)
        )

        in_channels_down5 = out_channels_down4 # 256
        out_channels_down5 = in_channels_down5*2 # 512
        self.down5 = nn.Sequential(
            ConvBlock(in_channels=in_channels_down5, out_channels = out_channels_down5),
            ConvBlock(in_channels=out_channels_down5, out_channels=out_channels_down5, stride=1, kernel_size=3)
        )

        in_channel_center = out_channels_down5 # 512
        out_channel_center = in_channel_center * 2 # 1024
        self.center_block = nn.Sequential(
            ConvBlock(in_channel_center, out_channel_center, 1, 3),
            ConvBlock(out_channel_center, out_channel_center, 1, 3),
            ConvBlock(out_channel_center, in_channel_center, 1, 3)
        )

        in_channel_up1 = in_channel_center + out_channels_down5 # 1024
        out_channel_up1 = in_channels_down5 # 256
        self.up1 = nn.Sequential(
            ConvTransBlock(in_channel_up1, out_channel_up1),
            ConvBlock(out_channel_up1, out_channel_up1, stride=1, kernel_size=3, padding=1)
        )

        in_channel_up2 = out_channel_up1 + out_channels_down4 # 512
        out_channel_up2 = in_channels_down4 #128
        self.up2 = nn.Sequential(
            ConvTransBlock(in_channel_up2, out_channel_up2),
            ConvBlock(out_channel_up2, out_channel_up2, stride=1, kernel_size=3, padding=1)
        )

        in_channel_up3 = out_channel_up2 + out_channels_down3 #256
        out_channel_up3 = in_channels_down3 #64
        self.up3 = nn.Sequential(
            ConvTransBlock(in_channel_up3, out_channel_up3),
            ConvBlock(out_channel_up3, out_channel_up3, stride=1, kernel_size=3, padding=1)
        )

        in_channel_up4 = out_channel_up3 + out_channels_down2 #128
        out_channel_up4 = in_channels_down2 #32
        self.up4 = nn.Sequential(
            ConvTransBlock(in_channel_up4, out_channel_up4),
            ConvBlock(out_channel_up4, out_channel_up4, stride=1, kernel_size=3, padding=1)
        )

        in_channel_up5 = out_channel_up4 + out_channels_down1 #64
        out_channel_up5 = out_channels #3
        self.up5 = nn.Sequential(
            ConvTransBlock(in_channel_up5, out_channel_up5),
            ConvBlock(out_channel_up5, out_channel_up5, stride=1, kernel_size=3, padding=1, activation='tanh')
        )

    def forward(self, x):
        #down
        out_down1 = self.down1(x)
        out_down2 = self.down2(out_down1)
        out_down3 = self.down3(out_down2)
        out_down4 = self.down4(out_down3)
        out_down5 = self.down5(out_down4)
        #up
        out_center = self.center_block(out_down5)
        out_up1 = self.up1(torch.cat([out_down5, out_center], 1))
        out_up2 = self.up2(torch.cat([out_down4, out_up1], 1))
        out_up3 = self.up3(torch.cat([out_down3, out_up2], 1))
        out_up4 = self.up4(torch.cat([out_down2, out_up3], 1))
        output = self.up5(torch.cat([out_down1, out_up4], 1))

        return output


