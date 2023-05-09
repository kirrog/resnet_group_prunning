import json
from functools import reduce

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, middle_channels=None):
        super(ResidualBlock, self).__init__()
        if middle_channels is None:
            middle_channels = out_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride=1)
        self.layer2 = self._make_layer(block, 256, layers[2], stride=1)
        self.layer3 = self._make_layer(block, 512, layers[3], stride=1)
        self.avgpool = nn.AvgPool2d(7)
        # self.fc = nn.Linear(4096, num_classes)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def calc_length(self, x):
        return reduce(lambda a, b: a * b, x.size())

    def recreate_layer(self, layer_seq, threshold: float):
        layers = [layer_seq[0]]
        layers_features = []
        for seq in layer_seq[1:]:
            input_conv_weight = seq.conv1[0].weight
            input_conv_bias = seq.conv1[0].bias
            input_norm_weight = seq.conv1[1].weight
            input_norm_bias = seq.conv1[1].bias
            output_conv_weight = seq.conv2[0].weight
            saved_features = []
            all_features = []
            for i in range(seq.conv1[0].weight.size()[0]):
                icw = torch.sum(torch.abs(input_conv_weight[i])) / self.calc_length(input_conv_weight[i])
                icb = torch.sum(torch.abs(input_conv_bias[i])) / self.calc_length(input_conv_weight[i])
                inw = torch.sum(torch.abs(input_norm_weight[i] - 1.0)) / self.calc_length(input_conv_weight[i])
                inb = torch.sum(torch.abs(input_norm_bias[i])) / self.calc_length(input_conv_weight[i])
                ocw = torch.sum(torch.abs(output_conv_weight[:, i])) / self.calc_length(input_conv_weight[i])
                value = icw + icb + inw + inb + ocw
                all_features.append((i, float(value)))
                if value > threshold:
                    saved_features.append((i, value))
            layers_features.append(all_features)
            if len(saved_features) == 0:
                continue
            in_size = input_conv_weight.size()
            out_size = output_conv_weight.size()
            input_conv_weight_new = torch.zeros((len(saved_features), in_size[1], in_size[2], in_size[3]))
            input_conv_bias_new = torch.zeros((len(saved_features)))
            input_norm_weight_new = torch.zeros((len(saved_features)))
            input_norm_bias_new = torch.zeros((len(saved_features)))
            output_conv_weight_new = torch.zeros((out_size[0], len(saved_features), out_size[2], out_size[3]))
            for j, pair in enumerate(saved_features):
                i, _ = pair
                input_conv_weight_new[j] = input_conv_weight[i]
                input_conv_bias_new[j] = input_conv_bias[i]
                input_norm_weight_new[j] = input_norm_weight[i]
                input_norm_bias_new[j] = input_norm_bias[i]
                output_conv_weight_new[:, j] = output_conv_weight[:, i]
            res_block = ResidualBlock(in_size[1], out_size[0], middle_channels=len(saved_features))
            res_block.conv1[0].weight = nn.parameter.Parameter(input_conv_weight_new)
            res_block.conv1[0].bias = nn.parameter.Parameter(input_conv_bias_new)
            res_block.conv1[1].weight = nn.parameter.Parameter(input_norm_weight_new)
            res_block.conv1[1].bias = nn.parameter.Parameter(input_norm_bias_new)
            res_block.conv2[0].weight = nn.parameter.Parameter(output_conv_weight_new)
            layers.append(res_block)
        self.recration_features.append(layers_features)
        return nn.Sequential(*layers)

    def recreation(self, threshold: float, path2dump_stats: str = "stat_dump.json"):
        self.recration_features = []
        self.layer0 = self.recreate_layer(self.layer0, threshold)
        self.layer1 = self.recreate_layer(self.layer1, threshold)
        with open(path2dump_stats, "w") as f:
            json.dump(self.recration_features, f)
