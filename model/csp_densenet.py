import torch

from block.csp_dense_block import csp_dense_block
from block.transition_block import transition_block

class csp_densenet(torch.nn.Module):

    def __init__(self, class_num=5, block_config=(6, 12, 24, 16), expansion_rate=4, growth_rate=32, droprate=0.2, activation=torch.nn.ReLU):

        super(csp_densenet, self).__init__()

        self.class_num = class_num
        self.block_config = block_config                ## filter
        self.growth_rate = growth_rate
        self.expansion_rate = expansion_rate


        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3,
                            out_channels=growth_rate * 2,
                            kernel_size=7,
                            stride=2,
                            padding=3,
                            bias=False)
        )

        self.pooling = torch.nn.MaxPool2d(3, stride=2, padding=1)

        inner_channels = growth_rate * 2
        num_layers = block_config[0]
        self.dense_layer1 = csp_dense_block(num_input_features=inner_channels,
                                                     num_layers=num_layers,
                                                     expansion_rate=self.expansion_rate,
                                                     growth_rate=self.growth_rate,
                                                     droprate=droprate,
                                                     part_ratio=0.5,
                                                     activation=activation)

        inner_channels = inner_channels + num_layers * growth_rate
        self.transition_layer1 = transition_block(in_channels=inner_channels,
                                                    out_channels=int(inner_channels / 2),
                                                    droprate=droprate,
                                                    activation=activation)

        inner_channels = int(inner_channels / 2)
        num_layers = block_config[1]
        self.dense_layer2 = csp_dense_block(num_input_features=inner_channels,
                                                     num_layers=num_layers,
                                                     expansion_rate=self.expansion_rate,
                                                     growth_rate=self.growth_rate,
                                                     droprate=droprate,
                                                     part_ratio=0.5,
                                                     activation=activation)
        inner_channels = inner_channels + num_layers * growth_rate
        self.transition_layer2 = transition_block(in_channels=inner_channels,
                                                     out_channels=int(inner_channels/ 2),
                                                     droprate=droprate,
                                                     activation=activation)
        inner_channels = int(inner_channels / 2)
        num_layers = block_config[2]
        self.dense_layer3 = csp_dense_block(num_input_features=inner_channels,
                                                     num_layers=num_layers,
                                                     expansion_rate=self.expansion_rate,
                                                     growth_rate=self.growth_rate,
                                                     droprate=droprate,
                                                     part_ratio=0.5,
                                                     activation=activation)
        inner_channels = inner_channels + num_layers * growth_rate
        self.transition_layer3 = transition_block(in_channels=inner_channels,
                                                     out_channels=int(inner_channels/ 2),
                                                     droprate=droprate,
                                                     activation=activation)
        inner_channels = int(inner_channels / 2)
        num_layers = block_config[3]
        self.dense_layer4 = csp_dense_block(num_input_features=inner_channels,
                                                     num_layers=num_layers,
                                                     expansion_rate=self.expansion_rate,
                                                     growth_rate=self.growth_rate,
                                                     droprate=droprate,
                                                     part_ratio=0.5,
                                                     activation=activation)
        inner_channels = inner_channels + num_layers * growth_rate
        self.classification_layer = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1),
                                                        torch.nn.Flatten(),
                                                        torch.nn.Linear(inner_channels,self.class_num),
                                                        torch.nn.Softmax(dim=1))

        # module 
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, torch.nn.BatchNorm2d):  # shifting param - scaling param (?)
                m.weight.data.fill_(1)  #
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pooling(x)
        x = self.dense_layer1(x)
        x = self.transition_layer1(x)
        x = self.dense_layer2(x)
        x = self.transition_layer2(x)
        x = self.dense_layer3(x)
        x = self.transition_layer3(x)
        x = self.dense_layer4(x)
        x = self.classification_layer(x)
        return x

def csp_densenet121(class_num=4, expansion_rate=4, growth_rate=32, droprate=0.2, activation=torch.nn.ReLU):
    block_config=(6,12,24,16)
    return csp_densenet(class_num=class_num, block_config=block_config, expansion_rate=expansion_rate, growth_rate=growth_rate, droprate=droprate, activation=activation)

def csp_densenet169(class_num=4, expansion_rate=4, growth_rate=32, droprate=0.2, activation=torch.nn.ReLU):
    block_config=(6,12,32,32)
    return csp_densenet(class_num=class_num, block_config=block_config, expansion_rate=expansion_rate, growth_rate=growth_rate, droprate=droprate, activation=activation)

def csp_densenet201(class_num=4, expansion_rate=4, growth_rate=32, droprate=0.2, activation=torch.nn.ReLU):
    block_config=(6,12,48,32)
    return csp_densenet(class_num=class_num, block_config=block_config, expansion_rate=expansion_rate, growth_rate=growth_rate, droprate=droprate, activation=activation)

def csp_densenet264(class_num=4, expansion_rate=4, growth_rate=32, droprate=0.2, activation=torch.nn.ReLU):
    block_config=(6,12,64,48)
    return csp_densenet(class_num=class_num, block_config=block_config, expansion_rate=expansion_rate, growth_rate=growth_rate, droprate=droprate, activation=activation)
