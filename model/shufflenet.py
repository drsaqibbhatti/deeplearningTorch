import torch
import torch.nn.functional as F
from block.shuffle_block import shuffle_block


class shufflenet(torch.nn.Module):
    def __init__(self, class_num=2, image_channel=3, groups=3, last_channel=576, block_config=[], activation=torch.nn.ReLU):
        super(shufflenet, self).__init__()

        self.groups = groups
        self.class_num = class_num

        self.stage1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=image_channel,
                            out_channels=24,
                            kernel_size=3,
                            stride=2,
                            bias=True,
                            padding=1),
            torch.nn.MaxPool2d(kernel_size=3,
                               stride=2,
                               padding=1)
        )

        self.stage2 = torch.nn.Sequential()
        self.stage3 = torch.nn.Sequential()
        self.stage4 = torch.nn.Sequential()
        self.classification_layer = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(last_channel, self.class_num),
            torch.nn.Softmax(dim=1)
        )

        blocks = [self.stage2, self.stage3, self.stage4]

        for block_index, (in_channel, out_channel, repeat) in enumerate(block_config):
            blocks[block_index].append(shuffle_block(
                in_channels=in_channel,
                out_channels=out_channel,
                groups=self.groups,
                grouped_conv=True,
                combine='concat',
                activation=activation))

            for repeat_index in range(repeat):
                blocks[block_index].append(shuffle_block(
                    in_channels=out_channel,
                    out_channels=out_channel,
                    groups=self.groups,
                    grouped_conv=True,
                    combine='add',
                    activation=activation))

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.classification_layer(x)
        return x

# block config
# in_channel, out_channel, repeat
def shufflenet_1(class_num=5,
             image_channel=3,
             activation=torch.nn.ReLU):
    groups=1
    block_config = (
        (24, 144, 3),
        (144, 288, 7),
        (288, 576, 3)
    )
    return shufflenet(class_num=class_num,
                  image_channel=image_channel,
                  last_channel=block_config[2][1],
                  groups=groups,
                  activation=activation,
                  block_config=block_config)

def shufflenet_2(class_num=5,
             image_channel=3,
             activation=torch.nn.ReLU):
    groups=2
    block_config = (
        (24, 200, 3),
        (200, 400, 7),
        (400, 800, 3)
    )
    return shufflenet(class_num=class_num,
                  image_channel=image_channel,
                  last_channel=block_config[2][1],
                  groups=groups,
                  activation=activation,
                  block_config=block_config)

def shufflenet_3(class_num=5,
             image_channel=3,
             activation=torch.nn.ReLU):
    groups=3
    block_config = (
        (24, 240, 3),
        (240, 480, 7),
        (480, 960, 3)
    )
    return shufflenet(class_num=class_num,
                  image_channel=image_channel,
                  last_channel=block_config[2][1],
                  groups=groups,
                  activation=activation,
                  block_config=block_config)

def shufflenet_4(class_num=5,
             image_channel=3,
             activation=torch.nn.ReLU):
    groups=4
    block_config = (
        (24, 272, 3),
        (272, 544, 7),
        (544, 1088, 3)
    )
    return shufflenet(class_num=class_num,
                  image_channel=image_channel,
                  last_channel=block_config[2][1],
                  groups=groups,
                  activation=activation,
                  block_config=block_config)

def shufflenet_5(class_num=5,
             image_channel=3,
             activation=torch.nn.ReLU):
    groups=8
    block_config = (
        (24, 384, 3),
        (384, 768, 7),
        (768, 1536, 3)
    )
    return shufflenet(class_num=class_num,
                  image_channel=image_channel,
                  last_channel=block_config[2][1],
                  groups=groups,
                  activation=activation,
                  block_config=block_config)