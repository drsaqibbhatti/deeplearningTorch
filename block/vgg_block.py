import torch

class vgg_block(torch.nn.Module):
    def __init__(self, in_channels, out_channel, kernel_size, stride, padding, activation=torch.nn.ReLU):
        super().__init__()
        self.vgg_block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels,
                            out_channels=out_channel,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding),
            torch.nn.BatchNorm2d(out_channel),
            activation()
        )

    def forward(self, x):
        x = self.vgg_block(x)
        return x