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
        result_images = []
        b = images.shape[0]
        for _, image in enumerate(images):
            h, w = images.shape[2:]
            h_range = torch.arange(h).to(self.device).type(torch.float32) / (h - 1)
            w_range = torch.arange(w).to(self.device).type(torch.float32) / (w - 1)

            h_channel = h_range.repeat(1, w, 1).transpose(1, 2)
            w_channel = w_range.repeat(1, h, 1)
            r_channel = torch.sqrt(torch.pow(h_channel - 0.5, 2) + torch.pow(w_channel - 0.5, 2))

            result_images.append(torch.cat((image, h_channel, w_channel, r_channel)))

        return self.conv(torch.stack(result_images))
