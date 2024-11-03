import torch
from block.ws_conv import ws_conv
from block.nf_residual_block import nf_residual_block
from block.nf_residualbottleneck_block import nf_residualbottleneck_block
from block.gamma_act_block import gamma_act_block


class nfresnet(torch.nn.Module):

    def __init__(self,
                 block_args=[],
                 class_num=5,
                 cardinality=32,
                 gap_dropout_probability=0.25,
                 stochastic_probability=0.25,
                 base_conv=nf_residual_block):
        super(nfresnet, self).__init__()

        self.class_num = class_num

        self.stem = torch.nn.Sequential(
            ws_conv(in_channels=3,
                     out_channels=64,
                     stride=2,
                     padding=3,
                     kernel_size=7,
                     bias=True),
            gamma_act_block(activation='relu',
                            inplace=True),
            torch.nn.MaxPool2d(kernel_size=3,
                               padding=1,
                               stride=2)
        )

        alpha = 0.2
        expected_std = 1.0
        blocks = []
        num_blocks = len(block_args)
        final_channel = 0
        for block_index, (in_dim, mid_dim, out_dim, stride) in enumerate(block_args):
            final_channel = out_dim
            beta = 1. / expected_std
            block_stochastic_probability = stochastic_probability * (block_index + 1) / num_blocks
            blocks.append(base_conv(in_dim=in_dim,
                                    mid_dim=mid_dim,
                                    out_dim=out_dim,
                                    stride=stride,
                                    beta=beta,
                                    alpha=alpha,
                                    groups=cardinality,
                                    stochastic_probability=block_stochastic_probability))
            if block_index == 0:
                expected_std = 1.0
            expected_std = (expected_std ** 2 + alpha ** 2) ** 0.5

        self.body = torch.nn.Sequential(*blocks)

        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.fc_dropout = torch.nn.Dropout2d(p=gap_dropout_probability)
        self.fc = torch.nn.Conv2d(in_channels=final_channel,
                                  out_channels=class_num,
                                  kernel_size=1)
        torch.nn.init.normal_(self.fc.weight, mean=0, std=0.01)

    def forward(self, x):
        x = self.stem(x)
        x = self.body(x)
        x = self.gap(x)
        x = self.fc_dropout(x)
        x = self.fc(x)
        x = x.view([-1, self.class_num])
        x = torch.softmax(x, dim=1)
        return x


def nfresnet18(class_num=5,
               gap_dropout_probability=0.25,
               stochastic_probability=0.25):
    block_args = (
        (64, 64, 64, 1),
        (64, 64, 64, 1),
        (64, 128, 128, 2),
        (128, 128, 128, 1),
        (128, 256, 256, 2),
        (256, 256, 256, 1),
        (256, 512, 512, 2),
        (512, 512, 512, 1),
    )
    return nfresnet(class_num=class_num,
                    cardinality=32,
                    gap_dropout_probability=gap_dropout_probability,
                    stochastic_probability=stochastic_probability,
                    block_args=block_args)


def nfresnet34(class_num=5,
               gap_dropout_probability=0.25,
               stochastic_probability=0.25):
    block_args = (
        (64, 64, 64, 1),
        (64, 64, 64, 1),
        (64, 64, 64, 1),
        (64, 128, 128, 2),
        (128, 128, 128, 1),
        (128, 128, 128, 1),
        (128, 256, 256, 2),
        (256, 256, 256, 1),
        (256, 256, 256, 1),
        (256, 256, 256, 1),
        (256, 256, 256, 1),
        (256, 256, 256, 1),
        (256, 512, 512, 2),
        (512, 512, 512, 1),
        (512, 512, 512, 1),
    )
    return nfresnet(class_num=class_num,
                    cardinality=32,
                    gap_dropout_probability=gap_dropout_probability,
                    stochastic_probability=stochastic_probability,
                    block_args=block_args)


def nfresnet50(class_num=5,
               gap_dropout_probability=0.25,
               stochastic_probability=0.25):
    block_args = (
        (64, 64, 256, 2),
        (256, 64, 256, 1),
        (256, 64, 256, 1),
        (256, 128, 512, 2),
        (512, 128, 512, 1),
        (512, 128, 512, 1),
        (512, 128, 512, 1),
        (512, 256, 1024, 2),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 512, 2048, 2),
        (2048, 512, 2048, 1),
        (2048, 512, 2048, 1),
    )
    return nfresnet(class_num=class_num,
                    cardinality=32,
                    gap_dropout_probability=gap_dropout_probability,
                    stochastic_probability=stochastic_probability,
                    block_args=block_args,
                    base_conv=nf_residualbottleneck_block)


def nfresnet101(class_num=5,
                gap_dropout_probability=0.25,
                stochastic_probability=0.25):
    block_args = (
        (64, 64, 256, 2),
        (256, 64, 256, 1),
        (256, 64, 256, 1),
        (256, 128, 512, 2),
        (512, 128, 512, 1),
        (512, 128, 512, 1),
        (512, 128, 512, 1),
        (512, 256, 1024, 2),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 512, 2048, 2),
        (2048, 512, 2048, 1),
        (2048, 512, 2048, 1),
    )
    return nfresnet(class_num=class_num,
                    cardinality=32,
                    gap_dropout_probability=gap_dropout_probability,
                    stochastic_probability=stochastic_probability,
                    block_args=block_args,
                    base_conv=nf_residualbottleneck_block)


def nfresnet152(class_num=5,
                gap_dropout_probability=0.25,
                stochastic_probability=0.25):
    block_args = (
        (64, 64, 256, 2),
        (256, 64, 256, 1),
        (256, 64, 256, 1),
        (256, 128, 512, 2),
        (512, 128, 512, 1),
        (512, 128, 512, 1),
        (512, 128, 512, 1),
        (512, 128, 512, 1),
        (512, 128, 512, 1),
        (512, 128, 512, 1),
        (512, 128, 512, 1),
        (512, 256, 1024, 2),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 512, 2048, 2),
        (2048, 512, 2048, 1),
        (2048, 512, 2048, 1),
    )
    return nfresnet(class_num=class_num,
                    cardinality=32,
                    gap_dropout_probability=gap_dropout_probability,
                    stochastic_probability=stochastic_probability,
                    block_args=block_args,
                    base_conv=nf_residualbottleneck_block)