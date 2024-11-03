import torch
from block.invertedbottleneck_block_v3 import invertedbottleneck_block_v3


class mobilenetv3_small(torch.nn.Module):

    def __init__(self, class_num=5, alpha=1.0, channel=1, activation=torch.nn.Hardswish):
        super(mobilenetv3_small, self).__init__()

        self.class_num = class_num
        self.channel = channel
        self.alpha = alpha

        self.stem = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=self.channel,
                            out_channels=16,
                            bias=False,
                            stride=2,
                            padding=1,
                            kernel_size=3),
            torch.nn.BatchNorm2d(16),
            activation(),
        )

        self.features = torch.nn.Sequential(
            invertedbottleneck_block_v3(in_channels=16, expansion_out=int(self.alpha*16), out_channels=int(self.alpha*16), stride=1, use_se=False, kernel_size=3, activation=torch.nn.ReLU),
            invertedbottleneck_block_v3(in_channels=int(self.alpha*16), expansion_out=int(self.alpha*64), out_channels=int(self.alpha*24), stride=2, use_se=False, kernel_size=3, activation=torch.nn.ReLU),
            invertedbottleneck_block_v3(in_channels=int(self.alpha*24), expansion_out=int(self.alpha*72), out_channels=int(self.alpha*24), stride=1, use_se=False, kernel_size=3, activation=torch.nn.ReLU),
            invertedbottleneck_block_v3(in_channels=int(self.alpha*24), expansion_out=int(self.alpha*72), out_channels=int(self.alpha*40), stride=2, use_se=True, kernel_size=5, activation=torch.nn.ReLU),
            invertedbottleneck_block_v3(in_channels=int(self.alpha*40), expansion_out=int(self.alpha*120), out_channels=int(self.alpha*40), stride=1, use_se=True, kernel_size=5, activation=torch.nn.ReLU),
            invertedbottleneck_block_v3(in_channels=int(self.alpha*40), expansion_out=int(self.alpha*120), out_channels=int(self.alpha*40), stride=1, use_se=True, kernel_size=5, activation=torch.nn.ReLU),
            invertedbottleneck_block_v3(in_channels=int(self.alpha*40), expansion_out=int(self.alpha*240), out_channels=int(self.alpha*80), stride=2, use_se=False, kernel_size=3, activation=activation),
            invertedbottleneck_block_v3(in_channels=int(self.alpha*80), expansion_out=int(self.alpha*200), out_channels=int(self.alpha*80), stride=1, use_se=False, kernel_size=3, activation=activation),
            invertedbottleneck_block_v3(in_channels=int(self.alpha*80), expansion_out=int(self.alpha*184), out_channels=int(self.alpha*80), stride=1, use_se=False, kernel_size=3, activation=activation),
            invertedbottleneck_block_v3(in_channels=int(self.alpha*80), expansion_out=int(self.alpha*184), out_channels=int(self.alpha*80), stride=1, use_se=False, kernel_size=3, activation=activation),
            invertedbottleneck_block_v3(in_channels=int(self.alpha*80), expansion_out=int(self.alpha*480), out_channels=int(self.alpha*112), stride=1, use_se=True, kernel_size=3, activation=activation),
            invertedbottleneck_block_v3(in_channels=int(self.alpha*112), expansion_out=int(self.alpha*672), out_channels=int(self.alpha*112), stride=1, use_se=True, kernel_size=3, activation=activation),
            invertedbottleneck_block_v3(in_channels=int(self.alpha*112), expansion_out=int(self.alpha*672), out_channels=int(self.alpha*160), stride=2, use_se=True, kernel_size=5, activation=activation),
            invertedbottleneck_block_v3(in_channels=int(self.alpha*160), expansion_out=int(self.alpha*960), out_channels=int(self.alpha*160), stride=1, use_se=True, kernel_size=5, activation=activation),
            invertedbottleneck_block_v3(in_channels=int(self.alpha*160), expansion_out=int(self.alpha*960), out_channels=int(self.alpha*160), stride=1, use_se=True, kernel_size=5, activation=activation),
            torch.nn.Conv2d(kernel_size=1,
                            bias=False,
                            in_channels=int(self.alpha*160),
                            out_channels=int(self.alpha*960)),
            torch.nn.BatchNorm2d(num_features=int(self.alpha*960)),
            torch.nn.Hardswish(),
            torch.nn.AdaptiveAvgPool2d(1),
        )

        self.fcn = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=1,
                            in_channels=int(self.alpha*960),
                            out_channels=1280,
                            bias=True),
            torch.nn.Hardswish(),
            torch.nn.Conv2d(kernel_size=1,
                            in_channels=1280,
                            out_channels=self.class_num,
                            bias=True)
        )

        # module -
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





class mobilenetv3_large(torch.nn.Module):

    def __init__(self, class_num=5, alpha=1.0, channel=1, activation=torch.nn.Hardswish):
        super(mobilenetv3_large, self).__init__()

        self.class_num = class_num
        self.channel = channel
        self.alpha = alpha

        self.stem = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=self.channel,
                            out_channels=16,
                            bias=False,
                            stride=2,
                            padding=1,
                            kernel_size=3),
            torch.nn.BatchNorm2d(16),
            activation(),
        )

        self.features = torch.nn.Sequential(
            invertedbottleneck_block_v3(in_channels=16, expansion_out=int(self.alpha*16), out_channels=int(self.alpha*16), stride=2, use_se=True, kernel_size=3, activation=torch.nn.ReLU),
            invertedbottleneck_block_v3(in_channels=int(self.alpha*16), expansion_out=int(self.alpha*72), out_channels=int(self.alpha*24), stride=2, use_se=False, kernel_size=3, activation=torch.nn.ReLU),
            invertedbottleneck_block_v3(in_channels=int(self.alpha*24), expansion_out=int(self.alpha*88), out_channels=int(self.alpha*24), stride=1, use_se=False, kernel_size=3, activation=torch.nn.ReLU),
            invertedbottleneck_block_v3(in_channels=int(self.alpha*24), expansion_out=int(self.alpha*96), out_channels=int(self.alpha*40), stride=2, use_se=True, kernel_size=5, activation=activation),
            invertedbottleneck_block_v3(in_channels=int(self.alpha*40), expansion_out=int(self.alpha*240), out_channels=int(self.alpha*40), stride=1, use_se=True, kernel_size=5, activation=activation),
            invertedbottleneck_block_v3(in_channels=int(self.alpha*40), expansion_out=int(self.alpha*240), out_channels=int(self.alpha*40), stride=1, use_se=True, kernel_size=5, activation=activation),
            invertedbottleneck_block_v3(in_channels=int(self.alpha*40), expansion_out=int(self.alpha*120), out_channels=int(self.alpha*48), stride=1, use_se=True, kernel_size=5, activation=activation),
            invertedbottleneck_block_v3(in_channels=int(self.alpha*48), expansion_out=int(self.alpha*144), out_channels=int(self.alpha*48), stride=1, use_se=True, kernel_size=5, activation=activation),
            invertedbottleneck_block_v3(in_channels=int(self.alpha*48), expansion_out=int(self.alpha*288), out_channels=int(self.alpha*96), stride=2, use_se=True, kernel_size=5, activation=activation),
            invertedbottleneck_block_v3(in_channels=int(self.alpha*96), expansion_out=int(self.alpha*576), out_channels=int(self.alpha*96), stride=1, use_se=True, kernel_size=5, activation=activation),
            invertedbottleneck_block_v3(in_channels=int(self.alpha*96), expansion_out=int(self.alpha*576), out_channels=int(self.alpha*96), stride=1, use_se=True, kernel_size=5, activation=activation),
            torch.nn.Conv2d(kernel_size=1,
                            bias=False,
                            in_channels=int(self.alpha*96),
                            out_channels=int(self.alpha*576)),
            torch.nn.BatchNorm2d(num_features=int(self.alpha*576)),
            torch.nn.Hardswish(),
            torch.nn.AdaptiveAvgPool2d(1),
        )

        self.fcn = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=1,
                            in_channels=int(self.alpha*576),
                            out_channels=1024,
                            bias=True),
            torch.nn.Hardswish(),
            torch.nn.Conv2d(kernel_size=1,
                            in_channels=1024,
                            out_channels=self.class_num,
                            bias=True)
        )

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


def mobilenetv3_small_alpha1_0(class_num=4, channel=1, activation=torch.nn.Hardswish):
    return mobilenetv3_small(class_num=class_num, channel=channel, alpha=1.0, activation=activation)

def mobilenetv3_small_alpha0_75(class_num=4, channel=1, activation=torch.nn.Hardswish):
    return mobilenetv3_small(class_num=class_num, channel=channel, alpha=0.75, activation=activation)

def mobilenetv3_large_alpha1_0(class_num=4, channel=1, activation=torch.nn.Hardswish):
    return mobilenetv3_large(class_num=class_num, channel=channel, alpha=1.0, activation=activation)

def mobilenetv3_large_alpha0_75(class_num=4, channel=1, activation=torch.nn.Hardswish):
    return mobilenetv3_large(class_num=class_num, channel=channel, alpha=0.75, activation=activation)
