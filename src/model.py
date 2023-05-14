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
        # print(torch.mean(out))
        out = self.conv2(out)
        # print(torch.mean(out))
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        # print(torch.mean(out))
        out = self.relu(out)
        # print(torch.mean(out))
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
        self.num_classes = num_classes

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

    def recreate_layer_with_filter_treshold(self, layer_seq, threshold: float):
        layers = [layer_seq[0]]
        layers_features = []
        for seq in layer_seq[1:]:
            input_conv_weight = seq.conv1[0].weight
            input_conv_bias = seq.conv1[0].bias
            input_norm_weight = seq.conv1[1].weight
            input_norm_bias = seq.conv1[1].bias
            input_norm_running_var = seq.conv1[1].running_var
            input_norm_running_mean = seq.conv1[1].running_mean
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
                if value >= threshold:
                    saved_features.append((i, value))
            layers_features.append(all_features)
            if len(saved_features) == 0:
                continue
            in_size = input_conv_weight.size()
            out_size = output_conv_weight.size()
            input_conv_weight_new_list = []
            input_conv_bias_new_list = []
            input_norm_weight_new_list = []
            input_norm_bias_new_list = []
            input_norm_running_mean_new_list = []
            input_norm_running_var_new_list = []
            output_conv_weight_new_list = []

            for j, pair in enumerate(saved_features):
                i, _ = pair
                input_conv_weight_new_list.append(input_conv_weight.data[i])
                input_conv_bias_new_list.append(input_conv_bias.data[i])
                input_norm_weight_new_list.append(input_norm_weight.data[i])
                input_norm_bias_new_list.append(input_norm_bias.data[i])
                input_norm_running_mean_new_list.append(input_norm_running_mean.data[i])
                input_norm_running_var_new_list.append(input_norm_running_var.data[i])
                output_conv_weight_new_list.append(output_conv_weight.data[:, i])

            input_conv_weight_new = nn.parameter.Parameter(torch.stack(input_conv_weight_new_list))
            input_conv_bias_new = nn.parameter.Parameter(torch.stack(input_conv_bias_new_list))
            input_norm_weight_new = nn.parameter.Parameter(torch.stack(input_norm_weight_new_list))
            input_norm_bias_new = nn.parameter.Parameter(torch.stack(input_norm_bias_new_list))
            input_norm_running_mean_new = nn.parameter.Parameter(torch.stack(input_norm_running_mean_new_list))
            input_norm_running_var_new = nn.parameter.Parameter(torch.stack(input_norm_running_var_new_list))
            output_conv_weight_new = nn.parameter.Parameter(torch.stack(output_conv_weight_new_list, dim=1))

            res_block = ResidualBlock(in_size[1], out_size[0], middle_channels=len(saved_features))
            # example = torch.rand((1, 64, 32, 32))
            # res_block(example)

            res_block.conv1[0].weight = input_conv_weight_new
            res_block.conv1[0].bias = input_conv_bias_new
            res_block.conv1[1].weight = input_norm_weight_new
            res_block.conv1[1].bias = input_norm_bias_new
            res_block.conv1[1].running_mean = input_norm_running_mean_new
            res_block.conv1[1].running_var = input_norm_running_var_new
            # res_block.conv2[0].weight = output_conv_weight_new

            # res_block.conv1[0].weight = input_conv_weight
            # res_block.conv1[0].bias = input_conv_bias

            # res_block.conv1[1].weight = input_norm_weight
            # res_block.conv1[1].bias = input_norm_bias

            res_block.conv1[1].num_batches_tracked = seq.conv1[1].num_batches_tracked
            # res_block.conv1[1].running_mean = input_norm_running_mean
            # res_block.conv1[1].running_var = input_norm_running_var
            res_block.conv1[1].training = False

            res_block.conv2[0].weight = output_conv_weight_new
            res_block.conv2[0].bias = seq.conv2[0].bias

            res_block.conv2[1].weight = seq.conv2[1].weight
            res_block.conv2[1].bias = seq.conv2[1].bias

            res_block.conv2[1].num_batches_tracked = seq.conv2[1].num_batches_tracked
            res_block.conv2[1].training = False
            res_block.conv2[1].running_mean = seq.conv2[1].running_mean
            res_block.conv2[1].running_var = seq.conv2[1].running_var

            # print("orig")
            # orig_res = seq(example)
            # print("new")
            # new_res = res_block(example)
            # compare = torch.sum(torch.abs(orig_res - new_res))
            # print(compare)
            layers.append(res_block)
        self.recreation_features.append(layers_features)
        return nn.Sequential(*layers)

    def calc_weights(self, seq):
        params = list(seq.parameters())
        return sum([torch.sum(torch.abs(x)) / reduce(lambda a, b: a * b, x.size()) for x in params])

    def calc_norm_weights(self, seq):
        params = list(seq.parameters())
        params[0] = params[0] - 1
        return sum([torch.sum(torch.abs(x)) / reduce(lambda a, b: a * b, x.size()) for x in params])

    def recreate_layer_with_block_threshold(self, layer_seq, threshold: float):
        layers = [layer_seq[0]]
        stats_data = []
        for seq in layer_seq[1:]:
            c1l = self.calc_weights(seq.conv1[0])
            c1b = self.calc_norm_weights(seq.conv1[1])
            c2l = self.calc_weights(seq.conv2[0])
            c2b = self.calc_norm_weights(seq.conv2[1])
            stats_data.append([c1l, c1b, c2l, c2b])
            sum_all = c1l + c1b + c2l + c2b
            # print(f"All: {sum_all} c1l: {c1l} c1b: {c1b} c2l: {c2l} c2b: {c2b}")
            if sum_all >= threshold:
                layers.append(seq)
        self.recreation_features.append(stats_data)
        return nn.Sequential(*layers)

    def recreation_with_filter_regularization(self, threshold: float):
        assert 1.0 >= threshold >= 0.0
        self.recreation_features = []
        with torch.no_grad():
            for lay in [self.layer0, self.layer1, self.layer2, self.layer3]:
                self.recreate_layer_with_filter_treshold(lay, 1.0)
            groups_values = []
            for x in self.recreation_features:
                for y in x:
                    for z in y:
                        groups_values.append(z[1])
            mx = max(groups_values)
            mn = min(groups_values)
            threshold = threshold * (mx - mn) + mn
            self.recreation_features = []
            self.layer0 = self.recreate_layer_with_filter_treshold(self.layer0, threshold)
            self.layer1 = self.recreate_layer_with_filter_treshold(self.layer1, threshold)
            self.layer2 = self.recreate_layer_with_filter_treshold(self.layer2, threshold)
            self.layer3 = self.recreate_layer_with_filter_treshold(self.layer3, threshold)

    def recreation_with_block_regularization(self, threshold: float):
        assert 1.0 >= threshold >= 0.0
        self.recreation_features = []
        with torch.no_grad():
            for lay in [self.layer0, self.layer1, self.layer2, self.layer3]:
                self.recreate_layer_with_block_threshold(lay, 1.0)
            groups_values = []
            for x in self.recreation_features:
                for y in x:
                    groups_values.append(sum(y))
            mx = max(groups_values)
            mn = min(groups_values)
            threshold = threshold * (mx - mn) + mn
            self.recreation_features = []
            self.layer0 = self.recreate_layer_with_block_threshold(self.layer0, threshold)
            self.layer1 = self.recreate_layer_with_block_threshold(self.layer1, threshold)
            self.layer2 = self.recreate_layer_with_block_threshold(self.layer2, threshold)
            self.layer3 = self.recreate_layer_with_block_threshold(self.layer3, threshold)
