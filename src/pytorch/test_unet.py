import torch.nn as nn
import random


from .models.unet import Unet

from .models.utils.unet import *
from .models.utils.model import get_flatten_pytorch_model
from .models.utils.test import FakeCmdargs


class Test(nn.Module):
    def __init__(self):
        super().__init__()
        n_channels = 3
        n_classes = 1000
        bilinear = False
        factor = 1

        self.layers = nn.ModuleList([
            DoubleConv(n_channels, 64),
            Down(64, 128),
            Down(128, 256),
            Down(256, 512),
            Down(512, 1024 // factor),
            Up(1024, 512 // factor, bilinear),
            Up(512, 256 // factor, bilinear),
            Up(256, 128 // factor, bilinear),
            Up(128, 64, bilinear),
            OutConv(64, n_classes),
        ])

if __name__ == '__main__':
    cmdargs = FakeCmdargs()
    cmdargs.smd = 'segment_cost_with_max'
    cmdargs.algo3 = True

    test = Test()

    print('\n\n')

    out = [
        layer
        for layer in test.modules()
        if not isinstance(layer, (Test, nn.Sequential, nn.ModuleList, DoubleConv, Down, Up, OutConv))
    ]
    out = [
        nn.ReLU(inplace=False) if isinstance(layer, nn.ReLU) else layer
        for layer in out
    ]
    for layer in out:
        if hasattr(layer, 'weight'):
            pass
            # print(layer.__class__.__name__, layer.weight.numel())
        print(layer)
        # if isinstance(layer, nn.BatchNorm2d):
        #     print('batchnorm', layer.num_features)

    # x = torch.rand(1, 3, 24, 24)
    #
    # dconv = DoubleConv(3, 64)
    # out = dconv(x)
    # print(out.size())
    #
    # x = torch.rand(1, 1024, 24, 24)
    #
    # convt = nn.ConvTranspose2d(1024, 256, kernel_size=2, stride=2)
    # print('up in_channels:', convt.in_channels)
    # print('up out_channels:', convt.out_channels)
    # print('up numel:', convt.weight.numel())
    # print('up kernel_size:', convt.kernel_size)
    #
    # out = convt(x)
    # print('size of out of convt(x):', out.size())
    #
    # x = torch.rand(1, 512, 24, 24)
    # down = Down(512, 1024)
    # out = down(x)
    # print('size of out of down(x):', out.size())
    # print(out.size()[1])


    # up = Up(1024, 512, bilinear=False)
    # out = up(out)



    # unet = UNet(cmdargs)

    # out = get_pytorch_model_layers_by_name('vgg19')
    #
    # for layer in out:
    #     if hasattr(layer, 'weight'):
    #         print('{} has weight'.format(layer))


    # conv2d = nn.Conv2d(
    #     in_channels=3,
    #     out_channels=100,
    #     kernel_size=3,
    # )
    #
    # conv2d = nn.Linear(
    #     in_features=100,
    #     out_features=400,
    # )
    #
    # size_single_w = conv2d.weight.element_size()
    # print(size_single_w)
    # data_point_w = conv2d.weight.numel()
    # print(data_point_w)
    #
    # print('conv2d has weight that occupies: {} bytes'.format(data_point_w))




