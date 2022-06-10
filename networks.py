import torch
import torch.nn as nn


def weights_init(m):
    '''
    randomly initialized from a Normal distribution with mean=0, stdev=0.02
    '''
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Flatten(nn.Module):
    def forward(self, x):
        N = x.shape[0]
        return x.view(N, -1)


class Generator_64(nn.Module):
    def __init__(self, z_channel, f_channel, g_channel):
        super(Generator_64, self).__init__()
        self.backbone = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(in_channels=z_channel, out_channels=f_channel * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(f_channel * 8),
            nn.ReLU(inplace=True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(in_channels=f_channel * 8, out_channels=f_channel * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(f_channel * 4),
            nn.ReLU(inplace=True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(in_channels=f_channel * 4, out_channels=f_channel * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(f_channel * 2),
            nn.ReLU(inplace=True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(in_channels=f_channel * 2, out_channels=f_channel, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(f_channel),
            nn.ReLU(inplace=True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(in_channels=f_channel, out_channels=g_channel, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, x):
        return self.backbone(x)


class Discriminator_64(nn.Module):
    def __init__(self, g_channel, f_channel):
        super(Discriminator_64, self).__init__()
        self.backbone = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(in_channels=g_channel, out_channels=f_channel, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(in_channels=f_channel, out_channels=f_channel * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(f_channel * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(in_channels=f_channel * 2, out_channels=f_channel * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(f_channel * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(in_channels=f_channel * 4, out_channels=f_channel * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(f_channel * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(in_channels=f_channel * 8, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.backbone(x)


class Generator_128(nn.Module):
    def __init__(self, z_channel, f_channel, g_channel):
        super(Generator_128, self).__init__()
        self.backbone = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(in_channels=z_channel, out_channels=f_channel * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(f_channel * 8),
            nn.ReLU(inplace=True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(in_channels=f_channel * 8, out_channels=f_channel * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(f_channel * 4),
            nn.ReLU(inplace=True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(in_channels=f_channel * 4, out_channels=f_channel * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(f_channel * 2),
            nn.ReLU(inplace=True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(in_channels=f_channel * 2, out_channels=f_channel, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(f_channel),
            nn.ReLU(inplace=True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(in_channels=f_channel, out_channels=f_channel, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(f_channel),
            nn.ReLU(inplace=True),
            # state size. (ngf) x 64 x 64
            nn.ConvTranspose2d(in_channels=f_channel, out_channels=g_channel, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # state size. (nc) x 128 x 128
        )

    def forward(self, x):
        return self.backbone(x)


class Discriminator_128(nn.Module):
    def __init__(self, g_channel, f_channel):
        super(Discriminator_128, self).__init__()
        self.backbone = nn.Sequential(
            # input is (nc) x 128 x 128
            nn.Conv2d(in_channels=g_channel, out_channels=f_channel, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 64
            nn.Conv2d(in_channels=f_channel, out_channels=f_channel, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(f_channel),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(in_channels=f_channel, out_channels=f_channel * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(f_channel * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(in_channels=f_channel * 2, out_channels=f_channel * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(f_channel * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(in_channels=f_channel * 4, out_channels=f_channel * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(f_channel * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(in_channels=f_channel * 8, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.backbone(x)


class Generator_128_v2(nn.Module):
    def __init__(self, z_channel, f_channel, g_channel):
        super(Generator_128_v2, self).__init__()
        self.backbone = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(in_channels=z_channel, out_channels=f_channel * 16, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(f_channel * 16),
            nn.ReLU(inplace=True),
            # state size. (ngf*16) x 4 x 4
            nn.ConvTranspose2d(in_channels=f_channel * 16, out_channels=f_channel * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(f_channel * 8),
            nn.ReLU(inplace=True),
            # state size. (ngf*8) x 8 x 8
            nn.ConvTranspose2d(in_channels=f_channel * 8, out_channels=f_channel * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(f_channel * 4),
            nn.ReLU(inplace=True),
            # state size. (ngf*4) x 16 x 16
            nn.ConvTranspose2d(in_channels=f_channel * 4, out_channels=f_channel * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(f_channel * 2),
            nn.ReLU(inplace=True),
            # state size. (ngf*2) x 32 x 32
            nn.ConvTranspose2d(in_channels=f_channel * 2, out_channels=f_channel, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(f_channel),
            nn.ReLU(inplace=True),
            # state size. (ngf) x 64 x 64
            nn.ConvTranspose2d(in_channels=f_channel, out_channels=g_channel, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # state size. (nc) x 128 x 128
        )

    def forward(self, x):
        return self.backbone(x)


class Discriminator_128_v2(nn.Module):
    def __init__(self, g_channel, f_channel):
        super(Discriminator_128_v2, self).__init__()
        self.backbone = nn.Sequential(
            # input is (nc) x 128 x 128
            nn.Conv2d(in_channels=g_channel, out_channels=f_channel, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 64
            nn.Conv2d(in_channels=f_channel, out_channels=f_channel * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(f_channel * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 32 x 32
            nn.Conv2d(in_channels=f_channel * 2, out_channels=f_channel * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(f_channel * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 16 x 16
            nn.Conv2d(in_channels=f_channel * 4, out_channels=f_channel * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(f_channel * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 8 x 8
            nn.Conv2d(in_channels=f_channel * 8, out_channels=f_channel * 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(f_channel * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*16) x 4 x 4
            nn.Conv2d(in_channels=f_channel * 16, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.backbone(x)


class Discriminator_64_PatchGAN(nn.Module):
    def __init__(self, g_channel, f_channel):
        super(Discriminator_64_PatchGAN, self).__init__()
        self.backbone = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(in_channels=g_channel, out_channels=f_channel, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(in_channels=f_channel, out_channels=f_channel * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(f_channel * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(in_channels=f_channel * 2, out_channels=f_channel * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(f_channel * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(in_channels=f_channel * 4, out_channels=f_channel * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(f_channel * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(in_channels=f_channel * 8, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.backbone(x)

class Discriminator_128_v2_PatchGAN(nn.Module):
    def __init__(self, g_channel, f_channel):
        super(Discriminator_128_v2_PatchGAN, self).__init__()
        self.backbone = nn.Sequential(
            # input is (nc) x 128 x 128
            nn.Conv2d(in_channels=g_channel, out_channels=f_channel, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 64
            nn.Conv2d(in_channels=f_channel, out_channels=f_channel * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(f_channel * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 32 x 32
            nn.Conv2d(in_channels=f_channel * 2, out_channels=f_channel * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(f_channel * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 16 x 16
            nn.Conv2d(in_channels=f_channel * 4, out_channels=f_channel * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(f_channel * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 8 x 8
            nn.Conv2d(in_channels=f_channel * 8, out_channels=f_channel * 16, kernel_size=4, stride=1, padding=1, bias=False), #here stride = 1
            nn.BatchNorm2d(f_channel * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*16) x 4 x 4
            nn.Conv2d(in_channels=f_channel * 16, out_channels=1, kernel_size=4, stride=1, padding=1, bias=False),
            #nn.Sigmoid()
        )
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.backbone(x)
        return self.pooling(x).view(x.size()[0], -1)


class Discriminator_128_v2_DC_PatchGAN(nn.Module):
    def __init__(self, g_channel, f_channel):
        super(Discriminator_128_v2_DC_PatchGAN, self).__init__()
        self.backbone = nn.Sequential(
            # input is (nc) x 128 x 128
            nn.Conv2d(in_channels=g_channel, out_channels=f_channel, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 64
            nn.Conv2d(in_channels=f_channel, out_channels=f_channel * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(f_channel * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 32 x 32
            nn.Conv2d(in_channels=f_channel * 2, out_channels=f_channel * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(f_channel * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 16 x 16
            nn.Conv2d(in_channels=f_channel * 4, out_channels=f_channel * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(f_channel * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 8 x 8
        )
        self.brunch1 = nn.Sequential(
            nn.Conv2d(in_channels=f_channel * 8, out_channels=f_channel * 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(f_channel * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*16) x 4 x 4
        )
        self.out1 = nn.Sequential(
            nn.Conv2d(in_channels=f_channel * 16, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),
            # state size. 1 x 1 x 1
            nn.Sigmoid()
        )
        self.brunch2 = nn.Sequential(
            nn.Conv2d(in_channels=f_channel * 8, out_channels=f_channel * 16, kernel_size=4, stride=1, padding=1, bias=False), #here stride = 1
            nn.InstanceNorm2d(f_channel * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*16) x 7 x 7
        )
        self.out2 = nn.Sequential(
            nn.Conv2d(in_channels=f_channel * 16, out_channels=1, kernel_size=4, stride=1, padding=1, bias=False),
            # state size. 1 x 6 x 6
            nn.AdaptiveAvgPool2d((1, 1))
            # state size. 1 x 1 x 1
        )

    def forward(self, x):
        x = self.backbone(x)
        x1 = self.out1(self.brunch1(x))
        x2 = self.out2(self.brunch2(x))
        return x1.view(-1), x2.view(-1)


class Discriminator_128_v2_DC_PatchGAN_CLS(Discriminator_128_v2_DC_PatchGAN):
    def __init__(self, g_channel, f_channel, num_classes):
        super(Discriminator_128_v2_DC_PatchGAN_CLS,self).__init__(g_channel, f_channel)
        self.pooling1 = nn.AdaptiveAvgPool2d((1, 1))
        self.pooling2 = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features=f_channel * 16 * 2, out_features=num_classes, bias=True)

    def forward(self, x):
        x = self.backbone(x)
        x1 = self.pooling1(self.brunch1(x)).view(x.size()[0], -1)
        x2 = self.pooling2(self.brunch2(x)).view(x.size()[0], -1)
        x = self.fc(torch.cat([x1, x2], dim=1))

        return x