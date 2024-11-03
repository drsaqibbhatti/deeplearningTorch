import torch
from block.residual_bottleneck_block import residual_bottleneck_block
from block.residual_block import residual_block


class resnet(torch.nn.Module):

    def __init__(self,
                 image_channel=3,
                 block_config=[],
                 last_channel=512,
                 class_num=5,
                 activation=torch.nn.ReLU,
                 base_conv=residual_block):
        super(resnet, self).__init__()

        self.class_num = class_num

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=image_channel,
                            out_channels=64,
                            stride=2,
                            padding=3,
                            kernel_size=7,
                            bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
        )

        self.conv2 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=3,
                               padding=1,
                               stride=2)
        )

        self.conv3 = torch.nn.Sequential()
        self.conv4 = torch.nn.Sequential()
        self.conv5 = torch.nn.Sequential()

        self.blocks = [self.conv2, self.conv3, self.conv4, self.conv5]

        for block_index, (in_channel, inner_channel, out_channel, stride, repeat) in enumerate(block_config):
            for repeat_index in range(repeat):
                if repeat_index == 0:
                    self.blocks[block_index].append(
                        base_conv(in_channels=in_channel, inner_channels=inner_channel, out_channels=out_channel, stride=stride))
                else:
                    self.blocks[block_index].append(
                        base_conv(in_channels=out_channel, inner_channels=inner_channel, out_channels=out_channel, stride=1))

        self.classification_layer = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(last_channel, self.class_num),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.classification_layer(x)
        return x

#block_config
#in, inner, out, stride, repeat (The stride option works only the first time.)

def resnet18(class_num=5,
             image_channel=3,
             activation=torch.nn.ReLU):
    block_config = (
        (64, 64, 64, 1, 2),
        (64, 128, 128, 2, 2),
        (128, 256, 256, 2, 2),
        (256, 512, 512, 2, 2)
    )
    return resnet(class_num=class_num,
                  image_channel=image_channel,
                  last_channel=block_config[3][2],
                  activation=activation,
                  block_config=block_config)


def resnet34(class_num=5,
             image_channel=3,
             activation=torch.nn.ReLU):
    block_config = (
        (64, 64, 64, 1, 3),
        (64, 128, 128, 2, 4),
        (128, 256, 256, 2, 6),
        (256, 512, 512, 2, 3)
    )
    return resnet(class_num=class_num,
                  image_channel=image_channel,
                  last_channel=block_config[3][2],
                  activation=activation,
                  block_config=block_config)


def resnet50(class_num=5,
             image_channel=3,
             activation=torch.nn.ReLU):
    block_config = (
        (64, 64, 256, 1, 3),
        (256, 128, 512, 2, 4),
        (512, 256, 1024, 2, 6),
        (1024, 512, 2048, 2, 3)
    )
    return resnet(class_num=class_num,
                  image_channel=image_channel,
                  last_channel=block_config[3][2],
                  activation=activation,
                  block_config=block_config,
                  base_conv=residual_bottleneck_block)


def resnet101(class_num=5,
             image_channel=3,
              activation=torch.nn.ReLU):
    block_config = (
        (64, 64, 256, 1, 3),
        (256, 128, 512, 2, 4),
        (512, 256, 1024, 2, 23),
        (1024, 512, 2048, 2, 3)
    )
    return resnet(class_num=class_num,
                  image_channel=image_channel,
                  last_channel=block_config[3][2],
                  activation=activation,
                  block_config=block_config,
                  base_conv=residual_bottleneck_block)


def resnet152(class_num=5,
             image_channel=3,
              activation=torch.nn.ReLU):
    block_config = (
        (64, 64, 256, 1, 3),
        (256, 128, 512, 2, 8),
        (512, 256, 1024, 2, 36),
        (1024, 512, 2048, 2, 3)
    )
    return resnet(class_num=class_num,
                  image_channel=image_channel,
                  last_channel=block_config[3][2],
                  activation=activation,
                  block_config=block_config,
                  base_conv=residual_bottleneck_block)