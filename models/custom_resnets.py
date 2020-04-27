"""
modified resnet architecture
no fc
no bn with tapering,no bn after down/upsample (down/upsample is not Sequiential),  track_running=false
maxpool??
addec conv2 at the end: number of output channels=n_classes
inplace=False for ReLUs : changed back to True
, bias=False
needs state_dict_utils.toggle_state_dict_resnets to toggle the weights in SL
"""

import torch
import torch.nn as nn
# from modules import customized_modules_simple as customized_modules
from modules import customized_modules_layerwise as customized_modules
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
Conv2d = customized_modules.AsymmetricFeedbackConv2d
ConvTranspose2d = customized_modules.AsymmetricFeedbackConvTranspose2d
Linear = customized_modules.LinearModule

#TODO: cifar modification and fixup
#TODO: inspect the output size on the backward net
#TODO: replace BN with initialization
#TODO: replace maxpool with conolution/downsamp/avgpool ?
#TODO: replace Avgpool with conv1x1
#TODO: omit fc
#TODO: toggle state dict

#TODO: get the forward model as a list and reveasymrese the list

__all__ = ['AsymResNet', 'asymresnet18', 'asymresnet34', 'asymresnet50', 'asymresnet101',
           'asymresnet152', 'asymresnext50_32x4d', 'asymresnext101_32x8d',
           'wide_asymresnet50_2', 'wide_asymresnet101_2']


model_urls = {
    'asymresnet18': 'https://download.pytorch.org/models/asymresnet18-5c106cde.pth',
    'asymresnet34': 'https://download.pytorch.org/models/asymresnet34-333f7ec4.pth',
    'asymresnet50': 'https://download.pytorch.org/models/asymresnet50-19c8e357.pth',
    'asymresnet101': 'https://download.pytorch.org/models/asymresnet101-5d3b4d8f.pth',
    'asymresnet152': 'https://download.pytorch.org/models/asymresnet152-b121ed2d.pth',
    'asymresnext50_32x4d': 'https://download.pytorch.org/models/asymresnext50_32x4d-7cdf4587.pth',
    'asymresnext101_32x8d': 'https://download.pytorch.org/models/asymresnext101_32x8d-8ba56ff5.pth',
    'wide_asymresnet50_2': 'https://download.pytorch.org/models/wide_asymresnet50_2-95faca4d.pth',
    'wide_asymresnet101_2': 'https://download.pytorch.org/models/wide_asymresnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, bias=False, algorithm='BP'):
    """3x3 convolution with padding"""
    return Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation, algorithm=algorithm)


def conv1x1(in_planes, out_planes, stride=1, bias=False, algorithm='BP'):
    """1x1 convolution"""
    return Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, algorithm=algorithm)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, algorithm='BP'):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride, bias=False, algorithm=algorithm)
        # self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes, bias=False, algorithm=algorithm)
        self.bn2 = norm_layer(planes, track_running_stats=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, algorithm='BP'):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width, bias=False, algorithm=algorithm)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation, bias=False, algorithm=algorithm)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion, bias=False, algorithm=algorithm)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class AsymResNet(nn.Module):

    def __init__(self, block, layers, n_classes=10,image_channels=3,base_channels=64, zero_init_asymresidual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, algorithm='BP'):
        super(AsymResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False, algorithm=algorithm)
        self.bn1 = norm_layer(self.inplanes, track_running_stats=False)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], algorithm=algorithm)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], algorithm=algorithm)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], algorithm=algorithm)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], algorithm=algorithm)
        self.conv2 = Conv2d(512, n_classes, kernel_size=3, stride=1, padding=1,
                               bias=False, algorithm=algorithm)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = Linear(512 * block.expansion, n_classes, algorithm=algorithm)

        for m in self.modules():
            if isinstance(m, Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each asymresidual branch,
        # so that the asymresidual branch starts with zeros, and each asymresidual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_asymresidual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, algorithm='BP'):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = conv1x1(self.inplanes, planes * block.expansion, stride, bias=False, algorithm=algorithm)
                # norm_layer(planes * block.expansion, track_running_stats=False),
            

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, algorithm=algorithm))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, algorithm=algorithm))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x) withiut maxpool the output is of size 14x14 instead of 7x7

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        latent = self.conv2(x)
        x = self.avgpool(latent)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)

        return latent, x.squeeze()

    def forward(self, x):
        return self._forward_impl(x)


def _asymresnet(arch, block, layers, pretrained, progasymress, **kwargs):
    model = AsymResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progasymress=progasymress)
        model.load_state_dict(state_dict)
    return model


def asymresnet18(pretrained=False, progasymress=True, **kwargs):
    r"""AsymResNet-18 model from
    `"Deep AsymResidual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progasymress (bool): If True, displays a progasymress bar of the download to stderr
    """
    return _asymresnet('asymresnet18', BasicBlock, [2, 2, 2, 2], pretrained, progasymress,
                   **kwargs)


def asymresnet34(pretrained=False, progasymress=True, **kwargs):
    r"""AsymResNet-34 model from
    `"Deep AsymResidual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progasymress (bool): If True, displays a progasymress bar of the download to stderr
    """
    return _asymresnet('asymresnet34', BasicBlock, [3, 4, 6, 3], pretrained, progasymress,
                   **kwargs)


def asymresnet50(pretrained=False, progasymress=True, **kwargs):
    r"""AsymResNet-50 model from
    `"Deep AsymResidual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progasymress (bool): If True, displays a progasymress bar of the download to stderr
    """
    return _asymresnet('asymresnet50', Bottleneck, [3, 4, 6, 3], pretrained, progasymress,
                   **kwargs)


def asymresnet101(pretrained=False, progasymress=True, **kwargs):
    r"""AsymResNet-101 model from
    `"Deep AsymResidual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progasymress (bool): If True, displays a progasymress bar of the download to stderr
    """
    return _asymresnet('asymresnet101', Bottleneck, [3, 4, 23, 3], pretrained, progasymress,
                   **kwargs)


def asymresnet152(pretrained=False, progasymress=True, **kwargs):
    r"""AsymResNet-152 model from
    `"Deep AsymResidual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progasymress (bool): If True, displays a progasymress bar of the download to stderr
    """
    return _asymresnet('asymresnet152', Bottleneck, [3, 8, 36, 3], pretrained, progasymress,
                   **kwargs)


def asymresnext50_32x4d(pretrained=False, progasymress=True, **kwargs):
    r"""AsymResNeXt-50 32x4d model from
    `"Aggregated AsymResidual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progasymress (bool): If True, displays a progasymress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _asymresnet('asymresnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progasymress, **kwargs)


def asymresnext101_32x8d(pretrained=False, progasymress=True, **kwargs):
    r"""AsymResNeXt-101 32x8d model from
    `"Aggregated AsymResidual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progasymress (bool): If True, displays a progasymress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _asymresnet('asymresnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progasymress, **kwargs)


def wide_asymresnet50_2(pretrained=False, progasymress=True, **kwargs):
    r"""Wide AsymResNet-50-2 model from
    `"Wide AsymResidual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as AsymResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in AsymResNet-50 has 2048-512-2048
    channels, and in Wide AsymResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progasymress (bool): If True, displays a progasymress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _asymresnet('wide_asymresnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progasymress, **kwargs)


def wide_asymresnet101_2(pretrained=False, progasymress=True, **kwargs):
    r"""Wide AsymResNet-101-2 model from
    `"Wide AsymResidual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as AsymResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in AsymResNet-50 has 2048-512-2048
    channels, and in Wide AsymResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progasymress (bool): If True, displays a progasymress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _asymresnet('wide_asymresnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progasymress, **kwargs)



"""
Backward 
"""

def convT3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, bias=False, algorithm='BP'):
    """3x3 convolution with padding"""
    return ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation, algorithm=algorithm)


def convT1x1(in_planes, out_planes, stride=1, bias=False, algorithm='BP'):
    """1x1 convolution"""
    return ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, algorithm=algorithm)


class BasicBlockT(nn.Module):
    expansion = 1
    __constants__ = ['upsample']

    def __init__(self, inplanes, planes, stride=1, upsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, algorithm='BP'):
        super(BasicBlockT, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlockT only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlockT")
        # Both self.conv1 and self.upsample layers upsample the input when stride != 1
        self.conv1 = convT3x3(planes, inplanes,  stride, bias=False, algorithm=algorithm)
        # self.bn1 = norm_layer(inplanes, track_running_stats=False)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = convT3x3(planes, planes, bias=False, algorithm=algorithm)
        self.bn2 = norm_layer(planes, track_running_stats=False)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv2(x)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv1(out)
        # out = self.bn1(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.relu(out)

        return out


class BottleneckT(nn.Module):
    expansion = 4
    __constants__ = ['upsample']

    def __init__(self, inplanes, planes, stride=1, upsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, algorithm='BP'):
        super(BottleneckT, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.upsample layers upsample the input when stride != 1
        self.conv1 = convT1x1(width, inplanes, bias=False, algorithm=algorithm)
        self.bn1 = norm_layer(width, track_running_stats=False)
        self.conv2 = convT3x3(width, width, stride, groups, dilation, bias=False, algorithm=algorithm)
        self.bn2 = norm_layer(width, track_running_stats=False)
        self.conv3 = convT1x1(width, planes * self.expansion, bias=False, algorithm=algorithm)
        self.bn3 = norm_layer(planes * self.expansion, track_running_stats=False)
        self.relu = nn.ReLU(inplace=False)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.relu(out)

        return out


class AsymResNetT(nn.Module):

    def __init__(self, block, layers, n_classes=10, image_channels=3, base_channels=64, zero_init_asymresidual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, algorithm='BP'):
        super(AsymResNetT, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64*8
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.conv2 = ConvTranspose2d(n_classes, 512, kernel_size=3, stride=1, padding=1,
                                bias=False, padding_mode='zeros', algorithm=algorithm)
        self.layer4 = self._make_layer(block, 256, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], algorithm=algorithm)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], algorithm=algorithm)
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], algorithm=algorithm)
        self.layer1 = self._make_layer(block, 64, layers[0], algorithm=algorithm)
        self.relu = nn.ReLU(inplace=False)
        self.bn1 = norm_layer(self.inplanes, track_running_stats=False)
        self.conv1 = ConvTranspose2d(self.inplanes, 3, kernel_size=3, stride=1, padding=1,
                               bias=False, algorithm=algorithm)


        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = Linear(512 * block.expansion, n_classes, algorithm=algorithm)

        for m in self.modules():
            if isinstance(m, ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each asymresidual branch,
        # so that the asymresidual branch starts with zeros, and each asymresidual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_asymresidual:
            for m in self.modules():
                if isinstance(m, BottleneckT):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlockT):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, algorithm='BP'):
        norm_layer = self._norm_layer
        upsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        
        layers = []
        layers.append(block(self.inplanes, self.inplanes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, algorithm=algorithm))
        

        if stride != 1 or self.inplanes != int(planes / block.expansion):
            upsample = convT1x1(self.inplanes, int(planes / block.expansion), stride, bias=False, algorithm=algorithm)
                # norm_layer(int(planes / block.expansion)),
            


        
        for _ in range(1, blocks):
            layers.append(block(planes, self.inplanes, stride, upsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, algorithm=algorithm))

        self.inplanes = int(planes / block.expansion)

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
       
        # x = torch.flatten(x, 1)
        # x = self.avgpool(x)
        x = self.conv2(x)

        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)

        # x = self.upsamp(x) # instead of maxpool in Fw

        x = self.relu(x)
        xbeforeconv1 = self.bn1(x)
        x = self.conv1(xbeforeconv1)

        return xbeforeconv1, x

    def forward(self, x):
        return self._forward_impl(x)


def _asymresnetT(arch, block, layers, pretrained, progress, **kwargs):
    model = AsymResNetT(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict) #TODO: toggle state_dict
    return model


def asymresnetT18(pretrained=False, progress=True, **kwargs):
    r"""AsymResNetT-18 model from
    `"Deep AsymResidual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _asymresnetT('asymresnetT18', BasicBlockT, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def asymresnetT34(pretrained=False, progress=True, **kwargs):
    r"""AsymResNetT-34 model from
    `"Deep AsymResidual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _asymresnetT('asymresnetT34', BasicBlockT, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def asymresnetT50(pretrained=False, progress=True, **kwargs):
    r"""AsymResNetT-50 model from
    `"Deep AsymResidual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _asymresnetT('asymresnetT50', BottleneckT, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def asymresnetT101(pretrained=False, progress=True, **kwargs):
    r"""AsymResNetT-101 model from
    `"Deep AsymResidual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _asymresnetT('asymresnetT101', BottleneckT, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def asymresnetT152(pretrained=False, progress=True, **kwargs):
    r"""AsymResNetT-152 model from
    `"Deep AsymResidual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _asymresnetT('asymresnetT152', BottleneckT, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def asymresnextT50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""AsymResNeXt-50 32x4d model from
    `"Aggregated AsymResidual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _asymresnetT('asymresnextT50_32x4d', BottleneckT, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def asymresnextT101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""AsymResNeXt-101 32x8d model from
    `"Aggregated AsymResidual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _asymresnetT('asymresnextT101_32x8d', BottleneckT, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_asymresnetT50_2(pretrained=False, progress=True, **kwargs):
    r"""Wide AsymResNetT-50-2 model from
    `"Wide AsymResidual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as AsymResNetT except for the BottleneckT number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in AsymResNetT-50 has 2048-512-2048
    channels, and in Wide AsymResNetT-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _asymresnetT('wide_asymresnetT50_2', BottleneckT, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_asymresnetT101_2(pretrained=False, progress=True, **kwargs):
    r"""Wide AsymResNetT-101-2 model from
    `"Wide AsymResidual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as AsymResNetT except for the BottleneckT number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in AsymResNetT-50 has 2048-512-2048
    channels, and in Wide AsymResNetT-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _asymresnetT('wide_asymresnetT101_2', BottleneckT, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)





