import torch
import torch.nn

from block.mobilev2_block import mobilev2_block
from block.mobilevit_block import mobilevit_block
from block.conv_bn_act_block import conv_bn_act_block

class mobilevit(torch.nn.Module):
    def __init__(self, image_size, block_config, last_channel, num_classes=5, patch_size=(2, 2), activation=torch.nn.SiLU):
        super().__init__()
        ih, iw, ic = image_size
        ph, pw = patch_size
        assert ih % ph == 0 and iw % pw == 0

        self.layer1 = torch.nn.Sequential()
        for block_name, arg1, arg2, arg3, arg4, arg5 in block_config[0]:
            if block_name == "conv":
                self.layer1.append(conv_bn_act_block(inp=arg1, oup=arg2, kernal_size=arg3, stride=arg4, activation=activation))
            elif block_name == "mv2":
                self.layer1.append(mobilev2_block(inp=arg1, oup=arg2, expansion=arg3, stride=arg4))
            elif block_name == "vit":
                self.layer1.append(mobilevit_block(channel=arg1, dim=arg2, mlp_dim=arg3, depth=arg4, kernel_size=arg5, patch_size=patch_size))

        self.layer2 = torch.nn.Sequential()
        for block_name, arg1, arg2, arg3, arg4, arg5 in block_config[1]:
            if block_name == "conv":
                self.layer2.append(conv_bn_act_block(inp=arg1, oup=arg2, kernal_size=arg3, stride=arg4, activation=activation))
            elif block_name == "mv2":
                self.layer2.append(mobilev2_block(inp=arg1, oup=arg2, expansion=arg3, stride=arg4))
            elif block_name == "vit":
                self.layer2.append(mobilevit_block(channel=arg1, dim=arg2, mlp_dim=arg3, depth=arg4, kernel_size=arg5, patch_size=patch_size))

        self.layer3 = torch.nn.Sequential()
        for block_name, arg1, arg2, arg3, arg4, arg5 in block_config[2]:
            if block_name == "conv":
                self.layer3.append(conv_bn_act_block(inp=arg1, oup=arg2, kernal_size=arg3, stride=arg4, activation=activation))
            elif block_name == "mv2":
                self.layer3.append(mobilev2_block(inp=arg1, oup=arg2, expansion=arg3, stride=arg4))
            elif block_name == "vit":
                self.layer3.append(mobilevit_block(channel=arg1, dim=arg2, mlp_dim=arg3, depth=arg4, kernel_size=arg5, patch_size=patch_size))

        self.layer4 = torch.nn.Sequential()
        for block_name, arg1, arg2, arg3, arg4, arg5 in block_config[3]:
            if block_name == "conv":
                self.layer4.append(conv_bn_act_block(inp=arg1, oup=arg2, kernal_size=arg3, stride=arg4, activation=activation))
            elif block_name == "mv2":
                self.layer4.append(mobilev2_block(inp=arg1, oup=arg2, expansion=arg3, stride=arg4))
            elif block_name == "vit":
                self.layer4.append(mobilevit_block(channel=arg1, dim=arg2, mlp_dim=arg3, depth=arg4, kernel_size=arg5, patch_size=patch_size))

        self.layer5 = torch.nn.Sequential()
        for block_name, arg1, arg2, arg3, arg4, arg5 in block_config[4]:
            if block_name == "conv":
                self.layer5.append(conv_bn_act_block(inp=arg1, oup=arg2, kernal_size=arg3, stride=arg4, activation=activation))
            elif block_name == "mv2":
                self.layer5.append(mobilev2_block(inp=arg1, oup=arg2, expansion=arg3, stride=arg4))
            elif block_name == "vit":
                self.layer5.append(mobilevit_block(channel=arg1, dim=arg2, mlp_dim=arg3, depth=arg4, kernel_size=arg5, patch_size=patch_size))

        self.classification_layer = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(last_channel, num_classes),
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


def mobilevit_xxs(image_size, num_label=4, activation=torch.nn.SiLU):
    expansion=2
    channels = [16, 24, 48, 64, 80, 320]

    #dim, mlp_dim, layer
    vit_config = [
        [64, 64*2, 2],
        [80, 80*4, 4],
        [96, 96*4, 3]
    ]

    # parameter
    # conv -    1:input_channel,    2:output_channel,   3:kernel_size,      4:stride,           5:-
    # mv2 -     1:input_channel,    2:output_channel,   3:expansion,        4:stride,           5:-
    # vit -     1:input_channel,    2:dim,              3:mlp_dim,          4:inner_fc_layer,   5:kernel_size
    block_config = [
        #layer1
        [
            ("conv", image_size[2], channels[0], 3, 2, 0),
            ("mv2", channels[0], channels[0], expansion, 1, 0)
        ],
        #layer2
        [
            ("mv2", channels[0], channels[1], expansion, 2, 0),
            ("mv2", channels[1], channels[1], expansion, 1, 0),
            ("mv2", channels[1], channels[1], expansion, 1, 0)
        ],
        #layer3
        [
            ("mv2", channels[1], channels[2], expansion, 2, 0),
            ("vit", channels[2], vit_config[0][0], vit_config[0][1], vit_config[0][2], 3)
        ],
        # layer4
        [
            ("mv2", channels[2], channels[3], expansion, 2, 0),
            ("vit", channels[3], vit_config[1][0], vit_config[1][1], vit_config[1][2], 3)
        ],
        # layer5
        [
            ("mv2", channels[3], channels[4], expansion, 2, 0),
            ("vit", channels[4], vit_config[2][0], vit_config[2][1], vit_config[2][2], 3),
            ("conv", channels[4], channels[5], 1, 1, 0)
        ]
    ]

    return mobilevit(image_size, block_config=block_config, last_channel=channels[5], num_classes=num_label, activation=activation)


def mobilevit_xs(image_size, num_label=4, activation=torch.nn.SiLU):
    expansion = 4
    channels = [16, 32, 48, 64, 80, 96, 384]

    # dim, mlp_dim, layer
    vit_config = [
        [96, 96 * 2, 2],
        [120, 120 * 4, 4],
        [144, 144 * 4, 3]
    ]

    # parameter
    # conv -    1:input_channel,    2:output_channel,   3:kernel_size,      4:stride,           5:-
    # mv2 -     1:input_channel,    2:output_channel,   3:expansion,        4:stride,           5:-
    # vit -     1:input_channel,    2:dim,              3:mlp_dim,          4:inner_fc_layer,   5:kernel_size
    block_config = [
        # layer1
        [
            ("conv", image_size[2], channels[0], 3, 2, 0),
            ("mv2", channels[0], channels[1], expansion, 1, 0)
        ],
        # layer2
        [
            ("mv2", channels[1], channels[2], expansion, 2, 0),
            ("mv2", channels[2], channels[2], expansion, 1, 0),
            ("mv2", channels[2], channels[2], expansion, 1, 0)
        ],
        # layer3
        [
            ("mv2", channels[2], channels[3], expansion, 2, 0),
            ("vit", channels[3], vit_config[0][0], vit_config[0][1], vit_config[0][2], 3)
        ],
        # layer4
        [
            ("mv2", channels[3], channels[4], expansion, 2, 0),
            ("vit", channels[4], vit_config[1][0], vit_config[1][1], vit_config[1][2], 3)
        ],
        # layer5
        [
            ("mv2", channels[4], channels[5], expansion, 2, 0),
            ("vit", channels[5], vit_config[2][0], vit_config[2][1], vit_config[2][2], 3),
            ("conv", channels[5], channels[6], 1, 1, 0)
        ]
    ]

    return mobilevit(image_size, block_config=block_config, last_channel=channels[6], num_classes=num_label,
                     activation=activation)


def mobilevit_s(image_size, num_label=4, activation=torch.nn.SiLU):
    expansion = 4
    channels = [16, 32, 64, 96, 128, 160, 640]

    # dim, mlp_dim, layer
    vit_config = [
        [114, 144 * 2, 2],
        [192, 192 * 4, 4],
        [240, 240 * 4, 3]
    ]

    # parameter
    # conv -    1:input_channel,    2:output_channel,   3:kernel_size,      4:stride,           5:-
    # mv2 -     1:input_channel,    2:output_channel,   3:expansion,        4:stride,           5:-
    # vit -     1:input_channel,    2:dim,              3:mlp_dim,          4:inner_fc_layer,   5:kernel_size
    block_config = [
        # layer1
        [
            ("conv", image_size[2], channels[0], 3, 2, 0),
            ("mv2", channels[0], channels[1], expansion, 1, 0)
        ],
        # layer2
        [
            ("mv2", channels[1], channels[2], expansion, 2, 0),
            ("mv2", channels[2], channels[2], expansion, 1, 0),
            ("mv2", channels[2], channels[2], expansion, 1, 0)
        ],
        # layer3
        [
            ("mv2", channels[2], channels[3], expansion, 2, 0),
            ("vit", channels[3], vit_config[0][0], vit_config[0][1], vit_config[0][2], 3)
        ],
        # layer4
        [
            ("mv2", channels[3], channels[4], expansion, 2, 0),
            ("vit", channels[4], vit_config[1][0], vit_config[1][1], vit_config[1][2], 3)
        ],
        # layer5
        [
            ("mv2", channels[4], channels[5], expansion, 2, 0),
            ("vit", channels[5], vit_config[2][0], vit_config[2][1], vit_config[2][2], 3),
            ("conv", channels[5], channels[6], 1, 1, 0)
        ]
    ]

    return mobilevit(image_size, block_config=block_config, last_channel=channels[6], num_classes=num_label,
                     activation=activation)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

