""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False,padding_mode="replicate"),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False,padding_mode="replicate"),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class SDoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, down=True):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.ModuleList([
            # nn.Conv2d(in_channels, mid_channels, kernel_size=(3 if down else 1), padding=(1 if down else 0)), # Add bias here
            nn.Conv2d(in_channels, mid_channels, kernel_size=(3 if down else 1), padding=(1 if down else 0), bias=True,
                      padding_mode="replicate"),
            # nn.BatchNorm2d(mid_channels),
             nn.LeakyReLU(inplace=True),
               # nn.LayerNorm(mid_channels),
            # nn.Conv2d(mid_channels, out_channels, kernel_size=(1 if down else 3), padding=(0 if down else 1)),
            nn.Conv2d(mid_channels, out_channels, kernel_size=(1 if down else 3), padding=(0 if down else 1), bias=True,
                      padding_mode="replicate"),
            # nn.BatchNorm2d(out_channels),
             nn.LeakyReLU(inplace=True)
            #    nn.LayerNorm(out_channels)
            ]
        )

    def forward(self, x):
        cnt = 0
        for layer in self.double_conv:
            cnt = cnt + 1
            x = layer(x)
            if cnt % 2 == 1:
                x = x.permute(0, 2, 3, 1)
            else:
                x = x.permute(0, 3, 1, 2)
        return x

class NSMDoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None, up=True):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=(3 if up else 1), padding=(1 if up else 0), bias=False,padding_mode="replicate"),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=(1 if up else 3), padding=(0 if up else 1), bias=False,padding_mode="replicate"),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            SDoubleConv(in_channels, out_channels, down=True)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class SDown(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.avgpool_conv = nn.Sequential(
            nn.AvgPool2d(2),
            SDoubleConv(in_channels, out_channels, down=True)
        )

    def forward(self, x):
        return self.avgpool_conv(x)
    

class DoubleSDown(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.avgpool_conv = nn.Sequential(
            nn.AvgPool2d(4),
            SDoubleConv(in_channels, out_channels, down=True)
        )

    def forward(self, x):
        return self.avgpool_conv(x)

class NSMDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.avgpool_conv = nn.Sequential(
            NSMDoubleConv(in_channels, out_channels, up=False),
            nn.AvgPool2d(2)
        )

    def forward(self, x):
        return self.avgpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class SUp(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = SDoubleConv(in_channels, out_channels, in_channels // 2, down=False)
        else:
            raise ValueError

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.conv(x1)
        # input is CHW
        # diffY = x2.size()[2] - x1.size()[2]
        # diffX = x2.size()[3] - x1.size()[3]
        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2])
        return x1 + x2
        # TODO: Sum then conv??
        
class DoubleSUp(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
            self.conv = SDoubleConv(in_channels, out_channels, in_channels // 2, down=False)
        else:
            raise ValueError

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.conv(x1)
        # input is CHW
        # diffY = x2.size()[2] - x1.size()[2]
        # diffX = x2.size()[3] - x1.size()[3]
        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2])
        return x1 + x2
        # TODO: Sum then conv??

class NSMUp(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False,padding_mode="replicate"),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, padding=0, bias=False,padding_mode="replicate"),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.conv1(x1)
        return self.conv2(x1 + x2)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1,padding_mode="replicate")

    def forward(self, x):
        return self.conv(x)
    
class Pixel2Channel(nn.Module): # For acceleration
    def __init__(self, inverse=False):
        super(Pixel2Channel, self).__init__()
    
    def forward(self, x):
        x = x.view(*x.shape[:2], x.shape[2]//2, 2, x.shape[3]//2, 2).permute(0, 2, 4, 3, 5, 1)
        x = x.reshape(*x.shape[:-3], -1).permute(0, 3, 1, 2)
        return x

class Channel2Pixel(nn.Module): # For acceleration
    def __init__(self):
        super(Pixel2Channel, self).__init__()
    
    def forward(self, x):
        assert x.shape[-1] == 4
        # Upscale ch 0
        base = F.interpolate(x[..., 0], scale_factor=2, mode='bilinear')
        # Mix ch 1, 2, 3
        x[..., 0] = 0
        # x = x.reshape()
        return base + x


""" Full assembly of the parts to form the complete network """

class NSM(nn.Module):
    def __init__(self, n_in_channels):
        super(NSM, self).__init__()
        self.pixel2channel = Pixel2Channel() # 4 -> 16
        self.down1 = NSMDown(16, 32)
        self.down2 = NSMDown(32, 64)
        self.down3 = NSMDown(64, 256)
        self.up1 = NSMUp(256, 64)
        self.up2 = NSMUp(64, 32)
        self.up3 = NSMUp(32, 4, mid_channels=16)
        self.channel2pixel = Pixel2Channel(inverse=True)
        pass

    def forward(self, x):
        x1 = self.pixel2channel(x) # 256*256, c16
        x2 = self.down1(x1) # 128*128, c32
        x3 = self.down2(x2) # 64*64, c64
        x4 = self.down3(x3) # 32*32, c256
        x = self.up1(x4, x3) # 64*64, c64
        x = self.up2(x, x2) # 128*128, c32
        x = self.up3(x, x1) # 256*256*4
        return self.channel2pixel(x) # 512*512*1


class SUNet(nn.Module):  # Shadow U-Net inspired from NSM. late 2022 implementation version.
    def __init__(self, n_in_channels, n_out_channels):
        super(SUNet, self).__init__()
        self.in_channels = n_in_channels
        self.out_channels = n_out_channels
        # initial version
        # self.inc = (SDoubleConv(n_in_channels, 16))
        # self.down1 = (SDown(16, 32))
        # self.down2 = (SDown(32, 64))
        # self.down3 = (SDown(64, 256))
        # self.up1 = (SUp(256, 64))
        # self.up2 = (SUp(64, 32))
        # self.up3 = (SUp(32, 16))
        # self.outc = (OutConv(16, n_out_channels))

        # half-channel size & pixel2channel
        self.inc = nn.Conv2d(n_in_channels, n_out_channels, kernel_size=1, padding=0, bias=True,
                      padding_mode="replicate")
        self.down1 = (SDown(n_out_channels, n_out_channels *2))
        self.down2 = (SDown(n_out_channels*2, n_out_channels*4))
        self.down3 = (SDown(n_out_channels*4, n_out_channels*8))
        self.down4 = (SDown(n_out_channels*8, n_out_channels*16))
        # self.down4 = (SDown(n_out_channels*16, n_out_channels*32))  # TODO: eliminate this layer?
        # self.up0 = (SUp(n_out_channels*32, n_out_channels*16))
        self.up0 = (SUp(n_out_channels*16, n_out_channels*8))
        self.up1 = (SUp(n_out_channels*8, n_out_channels*4))
        self.up2 = (SUp(n_out_channels*4, n_out_channels*2))
        self.up3 = (SUp(n_out_channels*2, n_out_channels))
        self.outc = (OutConv(n_out_channels, n_out_channels))
        # self.inc = (SDoubleConv(n_in_channels, n_out_channels))
        # self.down2 = (SDown(n_out_channels * 4, n_out_channels * 8))
        # self.down3 = (SDown(n_out_channels * 8, n_out_channels * 16))
        # self.down4 = (SDown(n_out_channels * 16, n_out_channels * 32))  # TODO: eliminate this layer?
        # self.up0 = (SUp(n_out_channels * 32, n_out_channels * 16))
        # self.up1 = (SUp(n_out_channels * 16, n_out_channels * 8))
        # self.up2 = (SUp(n_out_channels * 8, n_out_channels * 4))
        # self.outc = (OutConv(n_out_channels, n_out_channels))
    def forward(self, x):
        B,C,W,H = x.shape
        x1 = self.inc(x)  # TODO: pixel2channel
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up0(x5, x4)
        x = self.up1(x, x3)
        x = self.up2(x, x2)
        x = self.up3(x,x1)
        return self.outc(x)  # TODO: pixel2channel



class NewSUNet(nn.Module):  # Shadow U-Net inspired from NSM. late 2022 implementation version.
    def __init__(self, n_in_channels, n_out_channels):
        super(NewSUNet, self).__init__()
        self.in_channels = n_in_channels
        self.out_channels = n_out_channels
        # initial version
        # self.inc = (SDoubleConv(n_in_channels, 16))
        # self.down1 = (SDown(16, 32))
        # self.down2 = (SDown(32, 64))
        # self.down3 = (SDown(64, 256))
        # self.up1 = (SUp(256, 64))
        # self.up2 = (SUp(64, 32))
        # self.up3 = (SUp(32, 16))
        # self.outc = (OutConv(16, n_out_channels))
        # half-channel size & pixel2channel
        self.inc = nn.Conv2d(n_in_channels, n_out_channels, kernel_size=1, padding=0, bias=True,
                      padding_mode="replicate")
        self.down1 = (SDown(n_out_channels, n_out_channels *2))
        self.down2 = (SDown(n_out_channels*2, n_out_channels*4))
        self.down3 = (DoubleSDown(n_out_channels*4, n_out_channels*8))
        self.down4 = (DoubleSDown(n_out_channels*8, n_out_channels*16))
        # self.down4 = (SDown(n_out_channels*16, n_out_channels*32))  # TODO: eliminate this layer?
        # self.up0 = (SUp(n_out_channels*32, n_out_channels*16))
        self.up0 = (DoubleSUp(n_out_channels*16, n_out_channels*8))
        self.up1 = (DoubleSUp(n_out_channels*8, n_out_channels*4))
        self.up2 = (SUp(n_out_channels*4, n_out_channels*2))
        self.up3 = (SUp(n_out_channels*2, n_out_channels))
        self.outc = (OutConv(n_out_channels, n_out_channels))
        # self.inc = (SDoubleConv(n_in_channels, n_out_channels))
        # self.down2 = (SDown(n_out_channels * 4, n_out_channels * 8))
        # self.down3 = (SDown(n_out_channels * 8, n_out_channels * 16))
        # self.down4 = (SDown(n_out_channels * 16, n_out_channels * 32))  # TODO: eliminate this layer?
        # self.up0 = (SUp(n_out_channels * 32, n_out_channels * 16))
        # self.up1 = (SUp(n_out_channels * 16, n_out_channels * 8))
        # self.up2 = (SUp(n_out_channels * 8, n_out_channels * 4))
        # self.outc = (OutConv(n_out_channels, n_out_channels))
    def forward(self, x):
        B,C,W,H = x.shape
        x1 = self.inc(x)  # TODO: pixel2channel
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        print("x5 shape ",x5.shape)
        x = self.up0(x5, x4)
        x = self.up1(x, x3)
        x = self.up2(x, x2)
        x = self.up3(x,x1)
        return self.outc(x)  # TODO: pixel2channel
    
class DoubleSUNet(nn.Module):  # Shadow U-Net inspired from NSM. late 2022 implementation version.
    def __init__(self, n_in_channels, n_out_channels):
        super(DoubleSUNet, self).__init__()
        self.in_channels = n_in_channels
        self.out_channels = n_out_channels
        # initial version
        # self.inc = (SDoubleConv(n_in_channels, 16))
        # self.down1 = (SDown(16, 32))
        # self.down2 = (SDown(32, 64))
        # self.down3 = (SDown(64, 256))
        # self.up1 = (SUp(256, 64))
        # self.up2 = (SUp(64, 32))
        # self.up3 = (SUp(32, 16))
        # self.outc = (OutConv(16, n_out_channels))

        # half-channel size & pixel2channel
        self.inc = nn.Conv2d(n_in_channels, n_out_channels, kernel_size=1, padding=0, bias=True,
                      padding_mode="replicate")
        self.down1 = (DoubleSDown(n_out_channels, n_out_channels *2))
        self.down2 = (DoubleSDown(n_out_channels*2, n_out_channels*4))
        self.down3 = (DoubleSDown(n_out_channels*4, n_out_channels*8))
        self.down4 = (DoubleSDown(n_out_channels*8, n_out_channels*16))
        # self.down4 = (SDown(n_out_channels*16, n_out_channels*32))  # TODO: eliminate this layer?
        # self.up0 = (SUp(n_out_channels*32, n_out_channels*16))
        self.up0 = (DoubleSUp(n_out_channels*16, n_out_channels*8))
        self.up1 = (DoubleSUp(n_out_channels*8, n_out_channels*4))
        self.up2 = (DoubleSUp(n_out_channels*4, n_out_channels*2))
        self.up3 = (DoubleSUp(n_out_channels*2, n_out_channels))
        self.outc = (OutConv(n_out_channels, n_out_channels))
        # self.inc = (SDoubleConv(n_in_channels, n_out_channels))
        # self.down2 = (SDown(n_out_channels * 4, n_out_channels * 8))
        # self.down3 = (SDown(n_out_channels * 8, n_out_channels * 16))
        # self.down4 = (SDown(n_out_channels * 16, n_out_channels * 32))  # TODO: eliminate this layer?
        # self.up0 = (SUp(n_out_channels * 32, n_out_channels * 16))
        # self.up1 = (SUp(n_out_channels * 16, n_out_channels * 8))
        # self.up2 = (SUp(n_out_channels * 8, n_out_channels * 4))
        # self.outc = (OutConv(n_out_channels, n_out_channels))
    def forward(self, x):
        B,C,W,H = x.shape
        x1 = self.inc(x)  # TODO: pixel2channel
        # x2 = self.down1(x1)
        # x3 = self.down2(x2)
        # x4 = self.down3(x3)
        # x5 = self.down4(x4)
        # print(x5.shape)
        # x = self.up0(x5, x4)
        # x = self.up1(x, x3)
        # x = self.up2(x, x2)
        # x = self.up3(x,x1)
        return self.outc(x1)  # TODO: pixel2channel
    
    
    

    
    

class SUPNet(nn.Module):  # Shadow U-Net inspired from NSM. late 2022 implementation version.
    def __init__(self, n_in_channels, n_out_channels):
        super(SUPNet, self).__init__()
        # initial version
        # self.inc = (SDoubleConv(n_in_channels, 16))
        # self.down1 = (SDown(16, 32))
        # self.down2 = (SDown(32, 64))
        # self.down3 = (SDown(64, 256))
        # self.up1 = (SUp(256, 64))
        # self.up2 = (SUp(64, 32))
        # self.up3 = (SUp(32, 16))
        # self.outc = (OutConv(16, n_out_channels))

        # half-channel size & pixel2channel
        self.inc = (SDoubleConv(n_in_channels, n_out_channels))
        self.down1 = (SDown(n_out_channels, n_out_channels * 2))
        self.down2 = (SDown(n_out_channels * 2, n_out_channels * 4))
        self.down3 = (SDown(n_out_channels * 4, n_out_channels * 8))
        self.up0 = (SUp(n_out_channels * 8, n_out_channels * 4))
        self.up1 = (SUp(n_out_channels * 4, n_out_channels * 2))
        self.up2 = (SUp(n_out_channels * 2, n_out_channels))
        self.outc = (OutConv(n_out_channels, n_out_channels))

    def forward(self, x):
        x1 = self.inc(x)  # TODO: pixel2channel
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up0(x4, x3)
        x = self.up1(x, x2)
        x = self.up2(x, x1)
        return self.outc(x)  # TODO: pixel2channel



class SUPNetLow(nn.Module):  # Shadow U-Net inspired from NSM. late 2022 implementation version.
    def __init__(self, n_in_channels, n_out_channels):
        super(SUPNetLow, self).__init__()
        # initial version
        # self.inc = (SDoubleConv(n_in_channels, 16))
        # self.down1 = (SDown(16, 32))
        # self.down2 = (SDown(32, 64))
        # self.down3 = (SDown(64, 256))
        # self.up1 = (SUp(256, 64))
        # self.up2 = (SUp(64, 32))
        # self.up3 = (SUp(32, 16))
        # self.outc = (OutConv(16, n_out_channels))

        # half-channel size & pixel2channel
        self.inc = (SDoubleConv(n_in_channels, n_out_channels))
        self.down1 = (SDown(n_out_channels, n_out_channels * 2))
        self.down2 = (SDown(n_out_channels * 2, n_out_channels * 4))
        self.down3 = (SDown(n_out_channels * 4, n_out_channels * 8))
        self.up0 = (SUp(n_out_channels * 8, n_out_channels * 4))
        self.outc = (OutConv(n_out_channels * 4, n_out_channels*2))
        self.outc2 = (OutConv(n_out_channels * 2, n_out_channels))
        self.low  = torch.nn.UpsamplingBilinear2d(scale_factor=0.5)
    def forward(self, x):
        # x1 = self.inc(x)  # TODO: pixel2channel
        # x2 = self.down1(x1)
        # x3 = self.down2(x2)
        # x4 = self.down3(x3)
        # x = self.up0(x4, x3)
        # x = self.outc(x)
        # x = self.outc2(x)
        # x = self.low(x)
        return self.low(x) # TODO: pixel2channel


class SUNetEncoder(nn.Module):  # Shadow U-Net inspired from NSM. late 2022 implementation version.
    def __init__(self, n_in_channels,n_out_channels,level_cnt=3):
        super().__init__()
        self.in_channels = n_in_channels
        self.out_channels = n_out_channels
        self.inc = (SDoubleConv(n_in_channels, n_out_channels))
        self.downList = nn.ModuleList()
        for i in range(level_cnt):
            self.downList.append(SDown(n_out_channels, n_out_channels *2))
            n_out_channels = n_out_channels * 2

    def forward(self, x, VQVAE = False):
        B,C,W,H = x.shape
        x1 = self.inc(x)  # TODO: pixel2channel
        for layer in self.downList:
            x1 = layer(x1)
        return x1

class SUNetEncoder(nn.Module):  # Shadow U-Net inspired from NSM. late 2022 implementation version.
    def __init__(self, n_in_channels,n_out_channels,level_cnt=3):
        super().__init__()
        self.in_channels = n_in_channels
        self.out_channels = n_out_channels
        self.inc = (SDoubleConv(n_in_channels, n_out_channels))
        self.downList = nn.ModuleList()
        for i in range(level_cnt):
            
            self.downList.append(SDown(n_out_channels, n_out_channels *2))
            n_out_channels = n_out_channels * 2

    def forward(self, x, VQVAE = False):
        B,C,W,H = x.shape
        x1 = self.inc(x)  # TODO: pixel2channel
        for layer in self.downList:
            x1 = layer(x1)
        return x1

class SUNetSpecialEncoder(nn.Module):  # Shadow U-Net inspired from NSM. late 2022 implementation version.
    def __init__(self, n_in_channels,n_out_channels,level_cnt=3):
        super().__init__()
        self.in_channels = n_in_channels
        self.out_channels = n_out_channels
        self.inc = (SDoubleConv(n_in_channels, n_out_channels))
        self.downList = nn.ModuleList()
        for i in range(level_cnt):
            if i != 0:
                self.downList.append(DoubleSDown(n_out_channels, n_out_channels *2))
            else:
                self.downList.append(SDown(n_out_channels, n_out_channels *2))
            n_out_channels = n_out_channels * 2

    def forward(self, x, VQVAE = False):
        B,C,W,H = x.shape
        x1 = self.inc(x)  # TODO: pixel2channel
        feature_list = []
        feature_list.append(x1)
        for layer in self.downList:
            x1 = layer(x1)
            feature_list.append(x1)
        return feature_list


class SUNetDecoder(nn.Module):  # Shadow U-Net inspired from NSM. late 2022 implementation version.
    def __init__(self,  n_out_channels,n_final_channel):
        super(SUNet, self).__init__()
        self.up0 = (SUp(n_out_channels * 16, n_out_channels * 4))
        self.up1 = (SUp(n_out_channels * 4, n_out_channels * 2))
        self.up2 = (SUp(n_out_channels * 2, n_out_channels*1))
        self.outc = (OutConv(n_out_channels, n_final_channel))
    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        return x3
    

class SUNetSpecialDecoder(nn.Module):  # Shadow U-Net inspired from NSM. late 2022 implementation version.
    def __init__(self,  n_out_channels,n_final_channel):
        super(SUNetSpecialDecoder, self).__init__()
        self.up0 = (DoubleSUp(n_out_channels * 8, n_out_channels * 4))
        self.up1 = (DoubleSUp(n_out_channels * 4, n_out_channels * 2))
        self.up2 = (SUp(n_out_channels * 2, n_out_channels*1))
        self.outc = (OutConv(n_out_channels, n_final_channel))
    def forward(self, x_list):
        print(x_list[3].shape)
        x_list[2] = self.up0(x_list[3], x_list[2])
        x_list[1] = self.up1(x_list[2],x_list[1])
        x_list[0] = self.up2(x_list[1],x_list[0])
        output = self.outc(x_list[0])
        return output


class SUNetDecoder(nn.Module):  # Shadow U-Net inspired from NSM. late 2022 implementation version.
    def __init__(self,  n_out_channels,n_final_channel):
        super(SUNet, self).__init__()
        self.up0 = (SUp(n_out_channels * 16, n_out_channels * 4))
        self.up1 = (SUp(n_out_channels * 4, n_out_channels * 2))
        self.up2 = (SUp(n_out_channels * 2, n_out_channels*1))
        self.outc = (OutConv(n_out_channels, n_final_channel))
    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        return x3
class UNet(nn.Module):
    def __init__(self, n_channels, n_outchannel, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_outchannel = n_outchannel
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, n_channels))
        self.down1 = (Down(n_channels, n_channels*2))
        self.down2 = (Down(n_channels*2, n_channels*4))
        factor = 2 if bilinear else 1
        self.down3 = (Down(n_channels*4, n_channels*8 // factor))
        self.up1 = (Up(n_channels*8, n_channels*4//factor,bilinear))
        self.up2 = (Up(n_channels*4,n_channels*2//factor,bilinear))
        self.up3 = (Up(n_channels*2,n_channels//factor,bilinear))
        self.outc = (OutConv(n_channels, n_outchannel))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits
