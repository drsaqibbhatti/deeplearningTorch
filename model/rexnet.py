import torch
from block.rexnet_linearbottleneck_block import rexnet_linearbottleneck_block


class rexnetV1(torch.nn.Module):

    def __init__(self,
                 block_args=[],
                 class_num=5,
                 channel=1,
                 width_multple=1.0,
                 se_rate=12,
                 expansion_rate=6,
                 dropout_ratio=0.2,
                 input_channel=16,
                 final_channel=185):
        super(rexnetV1, self).__init__()

        self.class_num = class_num
        self.channel = channel
        self.width_multiple = width_multple

        self.stem_channel = int(32 * self.width_multiple) if self.width_multiple < 1.0 else 32
        self.inplanes = input_channel / self.width_multiple if self.width_multiple < 1.0 else input_channel

        self.channel_unit = 0
        self.layer_count = 0
        for block_index, (layers, stride, use_se) in enumerate(block_args):
            self.layer_count += layers
        self.layer_count -= 1
        self.channel_unit = (final_channel - self.inplanes) / self.layer_count



        self.stem = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=self.channel,
                            out_channels=self.stem_channel,
                            kernel_size=3,
                            bias=False,
                            stride=2,
                            padding=1),
            torch.nn.BatchNorm2d(num_features=self.stem_channel),
            torch.nn.SiLU(),
        )

        self.current_input_channel = self.stem_channel
        self.current_output_channel = self.inplanes

        modules = []
        self.final_channel = 0
        for block_index, (layers, stride, use_se) in enumerate(block_args):
            current_expansion_rate = expansion_rate if block_index != 0 else 1
            for index in range(layers):
                current_stride = stride if index == 0 and stride == 2 else 1
                modules.append(rexnet_linearbottleneck_block(in_channels=self.current_input_channel,
                                                             out_channels=round(self.current_output_channel * self.width_multiple),
                                                             stride=current_stride,
                                                             use_se=use_se,
                                                             expand_rate=current_expansion_rate,
                                                             se_rate=se_rate))
                self.current_input_channel = round(self.current_output_channel * self.width_multiple)
                self.current_output_channel += self.channel_unit
                self.current_output_channel = round(self.current_output_channel)


        print('final channel = ', self.current_output_channel)
        modules.append(torch.nn.Conv2d(kernel_size=1,
                                       in_channels=self.current_input_channel,
                                       out_channels=int(1280 * self.width_multiple),
                                       bias=False))
        modules.append(torch.nn.AdaptiveAvgPool2d(1))
        modules.append(torch.nn.SiLU())
        modules.append(torch.nn.Dropout2d(p=dropout_ratio))

        self.features = torch.nn.Sequential(torch.nn.Sequential(*modules))

        self.fcn = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=1,
                            in_channels=int(1280 * self.width_multiple),
                            out_channels=self.class_num,
                            bias=True)
        )

        # module
        self.initialize_weights()

    def initialize_weights(self):
        # track all layers
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)


    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.fcn(x)
        x = x.view([-1, self.class_num])
        x = torch.softmax(x, dim=1)
        return x


def rexnetV1_alpha1_0(class_num=4, channel=1):
    block_args = (
        (1, 1, False),
        (2, 2, False),
        (2, 2, True),
        (6, 2, True),
        (5, 2, True),
    )
    return rexnetV1(class_num=class_num, channel=channel, width_multple=1.0, block_args=block_args)


def rexnetV1_alpha0_75(class_num=4, channel=1):
    block_args = (
        (1, 1, False),
        (2, 2, False),
        (2, 2, True),
        (6, 2, True),
        (5, 2, True),
    )
    return rexnetV1(class_num=class_num, channel=channel, width_multple=0.75, block_args=block_args)
