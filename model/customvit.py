import torch
import torch.nn

from block.mobilev2_block import mobilev2_block
from block.mobilevit_block import mobilevit_block
from block.conv_bn_act_block import conv_bn_act_block

class customvit(torch.nn.Module):
    def __init__(self, image_size, block_config, last_channel, num_classes=5, patch_size=(2, 2), activation=torch.nn.SiLU):
        super().__init__()
        ih, iw, ic = image_size
        ph, pw = patch_size
        assert ih % ph == 0 and iw % pw == 0

        self.layer = torch.nn.Sequential()
        for block_name, arg1, arg2, arg3, arg4, arg5 in block_config:
            if block_name == "conv":
                self.layer.append(conv_bn_act_block(inp=arg1, oup=arg2, kernal_size=arg3, stride=arg4, activation=activation))
            elif block_name == "mv2":
                self.layer.append(mobilev2_block(inp=arg1, oup=arg2, expansion=arg3, stride=arg4))
            elif block_name == "vit":
                self.layer.append(mobilevit_block(channel=arg1, dim=arg2, mlp_dim=arg3, depth=arg4, kernel_size=arg5,patch_size=patch_size))

        self.classification_layer = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(last_channel, num_classes),
            torch.nn.Softmax(dim=1)
        )
    def forward(self, x):
        x = self.layer(x)
        x = self.classification_layer(x)
        return x


def customvit_xxs(image_size, num_label=4, activation=torch.nn.SiLU):
    expansion = 2
    channels = [8, 16, 24, 40]

    # dim, mlp_dim, layer
    vit_config = [
        [12, 12, 1],
        [16, 16, 2]
    ]

    # parameter
    # conv -    1:input_channel,    2:output_channel,   3:kernel_size,      4:stride,           5:-
    # mv2 -     1:input_channel,    2:output_channel,   3:expansion,        4:stride,           5:-
    # vit -     1:input_channel,    2:dim,              3:mlp_dim,          4:inner_fc_layer,   5:kernel_size
    block_config = [
        # layer1
        ("conv", image_size[2], channels[0], 3, 2, 0),
        ("conv", channels[0], channels[0], 3, 2, 0),
        ("conv", channels[0], channels[1], 3, 2, 0),
        ("conv", channels[1], channels[1], 3, 2, 0),
        ("mv2", channels[1], channels[2], expansion, 2, 0),
        ("vit", channels[2], vit_config[0][0], vit_config[0][1], vit_config[0][2], 3),
        ("mv2", channels[2], channels[2], expansion, 2, 0),
        ("vit", channels[2], vit_config[1][0], vit_config[1][1], vit_config[1][2], 3),
        ("conv", channels[2], channels[3], 1, 1, 0)
    ]

    return customvit(image_size, block_config=block_config, last_channel=channels[3], num_classes=num_label, activation=activation)

def customvit_xs(image_size, num_label=4, activation=torch.nn.SiLU):
    expansion = 2
    channels = [8, 19, 30, 100]

    # dim, mlp_dim, layer
    vit_config = [
        [12, 12*2, 1],
        [16, 16*2, 3],
        [8, 8*2, 2]
    ]

    # parameter
    # conv -    1:input_channel,    2:output_channel,   3:kernel_size,      4:stride,           5:-
    # mv2 -     1:input_channel,    2:output_channel,   3:expansion,        4:stride,           5:-
    # vit -     1:input_channel,    2:dim,              3:mlp_dim,          4:inner_fc_layer,   5:kernel_size
    block_config = [
        # layer1
        ("conv", image_size[2], channels[0], 3, 2, 0),
        ("conv", channels[0], channels[0], 3, 2, 0),
        ("conv", channels[0], channels[1], 3, 2, 0),
        ("mv2", channels[1], channels[1], expansion, 2, 0),
        ("vit", channels[1], vit_config[0][0], vit_config[0][1], vit_config[0][2], 3),
        ("mv2", channels[1], channels[2], expansion, 2, 0),
        ("vit", channels[2], vit_config[1][0], vit_config[1][1], vit_config[1][2], 3),
        ("mv2", channels[2], channels[2], expansion, 2, 0),
        ("vit", channels[2], vit_config[2][0], vit_config[2][1], vit_config[2][2], 3),
        ("conv", channels[2], channels[3], 1, 1, 0)
    ]

    return customvit(image_size, block_config=block_config, last_channel=channels[3], num_classes=num_label, activation=activation)

def customvit_s(image_size, num_label=4, activation=torch.nn.SiLU):
    expansion = 2
    channels = [16, 24, 32, 128]

    # dim, mlp_dim, layer
    vit_config = [
        [12, 12*2, 2],
        [16, 16*4, 4],
        [8, 8*4, 3]
    ]

    # parameter
    # conv -    1:input_channel,    2:output_channel,   3:kernel_size,      4:stride,           5:-
    # mv2 -     1:input_channel,    2:output_channel,   3:expansion,        4:stride,           5:-
    # vit -     1:input_channel,    2:dim,              3:mlp_dim,          4:inner_fc_layer,   5:kernel_size
    block_config = [
        # layer1
        ("conv", image_size[2], channels[0], 3, 2, 0),
        ("conv", channels[0], channels[0], 3, 2, 0),
        ("conv", channels[0], channels[1], 3, 2, 0),
        ("mv2", channels[1], channels[1], expansion, 2, 0),
        ("vit", channels[1], vit_config[0][0], vit_config[0][1], vit_config[0][2], 3),
        ("mv2", channels[1], channels[2], expansion, 2, 0),
        ("vit", channels[2], vit_config[1][0], vit_config[1][1], vit_config[1][2], 3),
        ("mv2", channels[2], channels[2], expansion, 2, 0),
        ("vit", channels[2], vit_config[2][0], vit_config[2][1], vit_config[2][2], 3),
        ("conv", channels[2], channels[3], 1, 1, 0)
    ]

    return customvit(image_size, block_config=block_config, last_channel=channels[3], num_classes=num_label, activation=activation)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

