# -*- coding: utf-8 -*-
# @Time    : 2019/4/16 17:13
# @Author  : MengnanChen
# @FileName: speech_embedder_net.py
# @Software: PyCharm
import math

import torch
import torch.nn as nn

from hparams import hparam as hp
from ge2e_speaker_vertification.utils import get_centroids, get_cossim, calc_loss


class SpeechEmbedder(nn.Module):
    def __init__(self):
        super(SpeechEmbedder, self).__init__()
        '''
        LSTM
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
            would mean stacking two LSTMs together to form a `stacked LSTM`,
            with the second LSTM taking in outputs of the first LSTM and
            computing the final results. Default: 1
        '''
        self.LSTM_stack = nn.LSTM(hp.data.nmels, hp.model.hidden, num_layers=hp.model.num_layer, batch_first=True)
        for name, param in self.LSTM_stack.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)
        self.projection = nn.Linear(hp.model.hidden, hp.model.proj)

    def forward(self, x):
        '''
        input x: [20,160,40]
        output x shape: [20,256]
        '''
        x, _ = self.LSTM_stack(x.float())  # [batch_size,frames,n_mels]
        x = x[:, x.size(1) - 1]  # only use last frame
        x = self.projection(x.float())
        x = x / torch.norm(x, dim=1).unsqueeze(1)
        return x


class ReLU(nn.Hardtanh):
    def __init__(self, inplace=False):
        super(ReLU, self).__init__(0, 20, inplace)

    def __repr__(self):
        inplace_str = 'inplace' if self.inplace else ''
        return '{}({})'.format(self.__class__.__name__, inplace_str)


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class DeepSpeakerResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(DeepSpeakerResNet, self).__init__()

        self.relu = ReLU(inplace=True)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, layers[0])

        self.inplanes = 128
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.layer2 = self._make_layer(block, 128, layers[1])
        self.inplanes = 256
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.layer3 = self._make_layer(block, 256, layers[2])
        self.inplanes = 512
        self.conv4 = nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.layer4 = self._make_layer(block, 512, layers[3])

        self.avgpool = nn.AdaptiveAvgPool2d((1, None))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        '''
        :param block: BasicBlock
        :param in_planes: in_planes of BasicBlock
        :param out_planes: out_planes of BasicBlock
        :param blocks: number of blocks in every block(out_Conv and inner_Conv)
        :param stride: stride of BasicBlock
        :return:
        '''
        layers = []
        layers.append(block(self.inplanes, planes, stride))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class DeepSpeakerEmbedder(nn.Module):
    def __init__(self, embedding_size=hp.model.proj, feature_dim=40):
        super(DeepSpeakerEmbedder, self).__init__()

        self.embedding_size = embedding_size

        self.model = DeepSpeakerResNet(BasicBlock, [1, 1, 1, 1])
        if feature_dim == 64:
            self.model.fc = nn.Linear(512 * 4, self.embedding_size)
        elif feature_dim == 40:
            self.model.fc = nn.Linear(512 * 3, self.embedding_size)

    def l2_norm(self, input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output

    def forward(self, x):

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.layer1(x)

        x = self.model.conv2(x)
        x = self.model.bn2(x)
        x = self.model.relu(x)
        x = self.model.layer2(x)

        x = self.model.conv3(x)
        x = self.model.bn3(x)
        x = self.model.relu(x)
        x = self.model.layer3(x)

        x = self.model.conv4(x)
        x = self.model.bn4(x)
        x = self.model.relu(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.model.fc(x)
        self.features = self.l2_norm(x)
        # Multiply by alpha = 10 as suggested in https://arxiv.org/pdf/1703.09507.pdf
        alpha = 10
        self.features = self.features * alpha

        return self.features


class GE2ELoss(nn.Module):
    def __init__(self, device):
        super(GE2ELoss, self).__init__()
        self.w = nn.Parameter(torch.tensor(10.).to(device), requires_grad=True)
        self.b = nn.Parameter(torch.tensor(-5.).to(device), requires_grad=True)
        self.device = device

    def forward(self, embeddings):
        torch.clamp(self.w, 1e-6)
        centroids = get_centroids(embeddings)
        cossim = get_cossim(embeddings, centroids)
        sim_matrix = self.w * cossim.to(self.device) + self.b
        loss, _ = calc_loss(sim_matrix)
        return loss

# class BasicBlock(nn.Module):
#     def __init__(self, in_planes, out_planes, stride=1, downsample=None):
#         super(BasicBlock, self).__init__()
#         self.conv1 = conv3x3(in_planes, out_planes, stride)
#         self.bn1 = nn.BatchNorm2d(out_planes)
#         self.relu = ReLU(inplace=True)
#         self.conv2 = conv3x3(out_planes, out_planes)
#         self.bn2 = nn.BatchNorm2d(out_planes)
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x):
#         residual = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         if self.downsample is not None:
#             residual = self.downsample(x)
#
#         out += residual
#         out = self.relu(out)
#
#         return out

# class DeepSpeakerResNet(nn.Module):
#     def __init__(self):
#         super(DeepSpeakerResNet, self).__init__()
#
#         self.block = BasicBlock
#         self.n_inner_block = hp.model.n_inner_res_blocks
#         self.layers_out_planes = hp.model.channels
#
#         self.relu = ReLU(inplace=True)
#
#         self.convs = []
#         self.bns = []
#         self.inner_layers = []
#
#         _layers_out_planes = [1] + self.layers_out_planes
#         for index in range(1, len(_layers_out_planes)):
#             self.convs.append(
#                 nn.Conv2d(_layers_out_planes[index - 1], _layers_out_planes[index], kernel_size=5, stride=2, padding=2,
#                           bias=False)
#             )
#             self.inner_layers.append(
#                 self._make_layer(self.block, in_planes=_layers_out_planes[index - 1],
#                                  out_planes=_layers_out_planes[index],
#                                  n_sub_blocks=self.n_inner_block[index - 1])
#             )
#             self.bns.append(nn.BatchNorm2d(_layers_out_planes[index]))
#
#         self.avgpool = nn.AdaptiveAvgPool2d((1, None))
#         self.fc = nn.Linear(_layers_out_planes[-1], hp.model.proj)
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#
#     def _make_layer(self, res_block, in_planes, out_planes, n_sub_blocks, stride=1):
#         '''
#         :param block: BasicBlock
#         :param in_planes: in_planes of BasicBlock
#         :param out_planes: out_planes of BasicBlock
#         :param blocks: number of blocks in every block(out_Conv and inner_Conv)
#         :param stride: stride of BasicBlock
#         :return:
#         '''
#         layers = []
#         for i in range(n_sub_blocks):
#             layers.append(res_block(in_planes, out_planes, stride))
#         return nn.Sequential(*layers)
#
#     def _l2_norm(self, inputs):
#         inputs_size = inputs.size()
#         buffer = torch.pow(inputs, 2)
#         normp = torch.sum(buffer, 1).add_(1e-10)
#         norm = torch.sqrt(normp)
#         _output = torch.div(inputs, norm.view(-1, 1).expand_as(inputs))
#         output = _output.view(inputs_size)
#         return output
#
#     def forward(self, x):
#         for conv, bn, inner_layer in zip(self.convs, self.bns, self.inner_layers):
#             x = conv(x)
#             x = bn(x)
#             x = self.relu(x)
#             x = inner_layer(x)
#
#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         x = self._l2_norm(x)
#         alpha = 10
#         x = x * alpha
#
#         return x
