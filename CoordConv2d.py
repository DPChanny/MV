import torch
import torch.nn as nn
import torch.nn.modules.conv as conv


class CoordConv2d(conv.Conv2d):
    def __init__(self, conv_2d):
        super(CoordConv2d, self).__init__(conv_2d.in_channels, conv_2d.out_channels, conv_2d.kernel_size,
                                          conv_2d.stride, conv_2d.padding, conv_2d.dilation,
                                          conv_2d.groups, False if conv_2d._parameters['bias'] is None else True)
        self.conv = nn.Conv2d(conv_2d.in_channels + 3, conv_2d.out_channels, conv_2d.kernel_size,
                              conv_2d.stride, conv_2d.padding, conv_2d.dilation,
                              conv_2d.groups, False if conv_2d._parameters['bias'] is None else True)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def forward(self, images):
        images = images.cpu()

        b, c, h, w = images.shape

        xx_range = torch.arange(h, dtype=torch.int32)[None, None, :, None]
        yy_range = torch.arange(w, dtype=torch.int32)[None, None, :, None]

        xx_channel = torch.matmul(xx_range, torch.ones((1, 1, 1, w), dtype=torch.int32))
        yy_channel = torch.matmul(yy_range, torch.ones((1, 1, 1, h), dtype=torch.int32))

        yy_channel = yy_channel.permute(0, 1, 3, 2)

        xx_channel = xx_channel.float() / (h - 1)
        yy_channel = yy_channel.float() / (w - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(b, 1, 1, 1)
        yy_channel = yy_channel.repeat(b, 1, 1, 1)

        rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2) + torch.pow(yy_channel - 0.5, 2))

        images = torch.cat((images, xx_channel, yy_channel, rr), dim=1)

        return self.conv(images.to(self.device))
