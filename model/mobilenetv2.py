import torch
from block.invertedbottleneck_block import invertedbottleneck_block


class mobilenetv2(torch.nn.Module):

    def __init__(self, class_num=5, alpha=1.0, channel=1, activation=torch.nn.ReLU6):
        super(mobilenetv2, self).__init__()
        self.class_num = class_num
        self.alpha = alpha
        self.channel = channel

        self.stem = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=self.channel,
                            out_channels=32,
                            bias=False,
                            stride=2,
                            padding=1,
                            kernel_size=3),
            torch.nn.BatchNorm2d(32),
            activation()
        )

        self.features = torch.nn.Sequential(
            invertedbottleneck_block(in_channels=32, out_channels=int(self.alpha * 16), expansion_rate=1, stride=1, activation=activation),
            invertedbottleneck_block(in_channels=int(self.alpha * 16), out_channels=int(self.alpha * 24), expansion_rate=6, stride=2, activation=activation),
            invertedbottleneck_block(in_channels=int(self.alpha * 24), out_channels=int(self.alpha * 24), expansion_rate=6, stride=1, activation=activation),
            invertedbottleneck_block(in_channels=int(self.alpha * 24), out_channels=int(self.alpha * 32), expansion_rate=6, stride=2, activation=activation),
            invertedbottleneck_block(in_channels=int(self.alpha * 32), out_channels=int(self.alpha * 32), expansion_rate=6, stride=1, activation=activation),
            invertedbottleneck_block(in_channels=int(self.alpha * 32), out_channels=int(self.alpha * 32), expansion_rate=6, stride=1, activation=activation),
            invertedbottleneck_block(in_channels=int(self.alpha * 32), out_channels=int(self.alpha * 64), expansion_rate=6, stride=2, activation=activation),
            invertedbottleneck_block(in_channels=int(self.alpha * 64), out_channels=int(self.alpha * 64), expansion_rate=6, stride=1, activation=activation),
            invertedbottleneck_block(in_channels=int(self.alpha * 64), out_channels=int(self.alpha * 64), expansion_rate=6, stride=1, activation=activation),
            invertedbottleneck_block(in_channels=int(self.alpha * 64), out_channels=int(self.alpha * 64), expansion_rate=6, stride=1, activation=activation),
            invertedbottleneck_block(in_channels=int(self.alpha * 64), out_channels=int(self.alpha * 96), expansion_rate=6, stride=1, activation=activation),
            invertedbottleneck_block(in_channels=int(self.alpha * 96), out_channels=int(self.alpha * 96), expansion_rate=6, stride=1, activation=activation),
            invertedbottleneck_block(in_channels=int(self.alpha * 96), out_channels=int(self.alpha * 96), expansion_rate=6, stride=1, activation=activation),
            invertedbottleneck_block(in_channels=int(self.alpha * 96), out_channels=int(self.alpha * 160), expansion_rate=6, stride=2, activation=activation),
            invertedbottleneck_block(in_channels=int(self.alpha * 160), out_channels=int(self.alpha * 160), expansion_rate=6, stride=1, activation=activation),
            invertedbottleneck_block(in_channels=int(self.alpha * 160), out_channels=int(self.alpha * 160), expansion_rate=6, stride=1, activation=activation),
            invertedbottleneck_block(in_channels=int(self.alpha * 160), out_channels=int(self.alpha * 320), expansion_rate=6, stride=1, activation=activation),
            torch.nn.Conv2d(in_channels=int(self.alpha * 320), out_channels=int(self.alpha * 1280), kernel_size=1, stride=1, bias=False),
            torch.nn.AdaptiveAvgPool2d(1),
        )

        self.fcn = torch.nn.Conv2d(in_channels=int(self.alpha * 1280),
                                   out_channels=self.class_num,
                                   kernel_size=1,
                                   bias=True)

        # module 
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, torch.nn.BatchNorm2d):  # shifting param, scaling param 
                m.weight.data.fill_(1)  #
                m.bias.data.zero_()


    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.fcn(x)
        x = x.view([-1, self.class_num])
        x = torch.softmax(x, dim=1)
        return x



def mobilenetv2_alpha0_25(class_num=4, channel=1, activation=torch.nn.ReLU6):
    return mobilenetv2(class_num=class_num, channel=channel, alpha=0.25, activation=activation)

def mobilenetv2_alpha0_5(class_num=4, channel=1, activation=torch.nn.ReLU6):
    return mobilenetv2(class_num=class_num, channel=channel, alpha=0.5, activation=activation)

def mobilenetv2_alpha0_75(class_num=4, channel=1, activation=torch.nn.ReLU6):
    return mobilenetv2(class_num=class_num, channel=channel, alpha=0.75, activation=activation)

def mobilenetv2_alpha1_0(class_num=4, channel=1, activation=torch.nn.ReLU6):
    return mobilenetv2(class_num=class_num, channel=channel, alpha=1.0, activation=activation)

def mobilenetv2_alpha1_4(class_num=4, channel=1, activation=torch.nn.ReLU6):
    return mobilenetv2(class_num=class_num, channel=channel, alpha=1.4, activation=activation)