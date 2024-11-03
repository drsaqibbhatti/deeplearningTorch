import torch
import torch.nn.functional as F
from block.vgg_block import vgg_block

class vgg(torch.nn.Module):

    def __init__(self, image_channel=3, last_channel=512, class_num=5, block_config=[], activation=torch.nn.ReLU):
        super(vgg, self).__init__()

        self.drop_rate = 0.3
        self.class_num = class_num

        self.layer1 = torch.nn.Sequential()
        self.layer2 = torch.nn.Sequential()
        self.layer3 = torch.nn.Sequential()
        self.layer4 = torch.nn.Sequential()
        self.layer5 = torch.nn.Sequential()

        layers = [self.layer1,self.layer2,self.layer3,self.layer4,self.layer5]

        for layer_index in range(5):
            for block_index, (in_channel, out_channel, kernel_size, repeat) in enumerate(block_config[layer_index]):
                for repeat_index in range(repeat):
                    if repeat_index == 0:
                        layers[layer_index].append(vgg_block(
                            in_channels=in_channel,
                            out_channel=out_channel,
                            kernel_size=kernel_size,
                            padding=kernel_size // 2,
                            stride=1))
                    else:
                        layers[layer_index].append(vgg_block(
                            in_channels=out_channel,
                            out_channel=out_channel,
                            kernel_size=kernel_size,
                            padding=kernel_size // 2,
                            stride=1))
            layers[layer_index].append(torch.nn.MaxPool2d(kernel_size=2, stride=2))

        self.classification_layer = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(last_channel, 4096),
            torch.nn.Linear(4096, 4096),
            torch.nn.Linear(4096, self.class_num),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.classification_layer(x)
        return x

#block_config
#in_channel, out_channel, kernel_size, repeat
def vgg_a(class_num=5,
             image_channel=3,
             activation=torch.nn.ReLU):
    block_config = (
        #layer1
        [
            (image_channel, 64, 3, 1)
        ],
        # layer2
        [
            (64, 128, 3, 1)
        ],
        # layer3
        [
            (128, 256, 3, 2)
        ],
        # layer4
        [
            (256, 512, 3, 2)
        ],
        # layer5
        [
            (512, 512, 3, 2)
        ],
    )
    return vgg(class_num=class_num,
                  image_channel=image_channel,
                  last_channel=block_config[4][0][1],
                  activation=activation,
                  block_config=block_config)

def vgg_b(class_num=5,
             image_channel=3,
             activation=torch.nn.ReLU):
    block_config = (
        #layer1
        [
            (image_channel, 64, 3, 2)
        ],
        # layer2
        [
            (64, 128, 3, 2)
        ],
        # layer3
        [
            (128, 256, 3, 2)
        ],
        # layer4
        [
            (256, 512, 3, 2)
        ],
        # layer5
        [
            (512, 512, 3, 2)
        ],
    )
    return vgg(class_num=class_num,
                  image_channel=image_channel,
                  last_channel=block_config[4][0][1],
                  activation=activation,
                  block_config=block_config)

def vgg_c(class_num=5,
             image_channel=3,
             activation=torch.nn.ReLU):
    block_config = (
        #layer1
        [
            (image_channel, 64, 3, 2)
        ],
        # layer2
        [
            (64, 128, 3, 2)
        ],
        # layer3
        [
            (128, 256, 3, 2),
            (256, 256, 1, 1)
        ],
        # layer4
        [
            (256, 512, 3, 2),
            (512, 512, 1, 1)
        ],
        # layer5
        [
            (512, 512, 3, 2),
            (512, 512, 1, 1)
        ],
    )
    return vgg(class_num=class_num,
                  image_channel=image_channel,
                  last_channel=block_config[4][1][1],
                  activation=activation,
                  block_config=block_config)

def vgg_d(class_num=5,
             image_channel=3,
             activation=torch.nn.ReLU):
    block_config = (
        #layer1
        [
            (image_channel, 64, 3, 2)
        ],
        # layer2
        [
            (64, 128, 3, 2)
        ],
        # layer3
        [
            (128, 256, 3, 3)
        ],
        # layer4
        [
            (256, 512, 3, 3)
        ],
        # layer5
        [
            (512, 512, 3, 3)
        ],
    )
    return vgg(class_num=class_num,
                  image_channel=image_channel,
                  last_channel=block_config[4][0][1],
                  activation=activation,
                  block_config=block_config)

def vgg_e(class_num=5,
             image_channel=3,
             activation=torch.nn.ReLU):
    block_config = (
        #layer1
        [
            (image_channel, 64, 3, 2)
        ],
        # layer2
        [
            (64, 128, 3, 2)
        ],
        # layer3
        [
            (128, 256, 3, 4)
        ],
        # layer4
        [
            (256, 512, 3, 4)
        ],
        # layer5
        [
            (512, 512, 3, 4)
        ],
    )
    return vgg(class_num=class_num,
                  image_channel=image_channel,
                  last_channel=block_config[4][0][1],
                  activation=activation,
                  block_config=block_config)