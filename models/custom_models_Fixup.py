import torch
import torch.nn as nn
import numpy as np
from modules import customized_modules
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
Conv2d = customized_modules.AsymmetricFeedbackConv2d 
ConvTranspose2d = customized_modules.AsymmetricFeedbackConvTranspose2d
Linear = customized_modules.LinearModule

__all__ = ['FixupResNet', 'fixup_resnet20', 'fixup_resnet32', 'fixup_resnet44', 'fixup_resnet56', 'fixup_resnet110', 'fixup_resnet1202']


def conv3x3(in_planes, out_planes, stride=1,algorithm='BP'):
    """3x3 convolution with padding"""
    return Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, algorithm=algorithm)

def conv1x1(in_planes, out_planes, stride=1, algorithm='BP'):
    """1x1 convolution"""
    return Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, algorithm=algorithm)


class Flatten(nn.Module):
    r"""
    Flattens a contiguous range of dims into a tensor. For use with :class:`~nn.Sequential`.
    Args:
        start_dim: first dim to flatten (default = 1).
        end_dim: last dim to flatten (default = -1).

    Shape:
        - Input: :math:`(N, *dims)`
        - Output: :math:`(N, \prod *dims)` (for the default case).


    Examples::
        >>> m = nn.Sequential(
        >>>     nn.Conv2d(1, 32, 5, 1, 1),
        >>>     nn.Flatten()
        >>> )
    """
    __constants__ = ['start_dim', 'end_dim']

    def __init__(self, start_dim=1, end_dim=-1):
        super(Flatten, self).__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input):
        return input.flatten(self.start_dim, self.end_dim)


class FixupBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,algorithm='BP'):
        super(FixupBasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.bias1a = nn.Parameter(torch.zeros(1))
        self.conv1 = conv3x3(inplanes, planes, stride, algorithm=algorithm)
        self.bias1b = nn.Parameter(torch.zeros(1))
        self.relu = nn.ReLU(inplace=False)
        self.bias2a = nn.Parameter(torch.zeros(1))
        self.conv2 = conv3x3(planes, planes, algorithm=algorithm)
        self.scale = nn.Parameter(torch.ones(1))
        self.bias2b = nn.Parameter(torch.zeros(1))
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x + self.bias1a)
        out = self.relu(out + self.bias1b)

        out = self.conv2(out + self.bias2a)
        out = out * self.scale + self.bias2b

        if self.downsample is not None:
            identity = self.downsample(x + self.bias1a)
            identity = torch.cat((identity, torch.zeros_like(identity)), 1)

        out += identity
        out = self.relu(out)

        return out


class FixupResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10,algorithm='BP'):
        super(FixupResNet, self).__init__()
        self.num_layers = sum(layers)
        self.inplanes = 16
        self.conv1 = conv3x3(1, 16, algorithm=algorithm)
        self.bias1 = nn.Parameter(torch.zeros(1))
        self.relu = nn.ReLU(inplace=False)
        self.layer1 = self._make_layer(block, 16, layers[0],algorithm=algorithm)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2,algorithm=algorithm)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2,algorithm=algorithm)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = Flatten()
        self.biasfc = nn.Parameter(torch.zeros(1))
        self.fc = nn.Linear(64, num_classes) #note the false bias in the original fixup was true

        for m in self.modules():
            if isinstance(m, FixupBasicBlock):
                nn.init.normal_(m.conv1.weight, mean=0, std=np.sqrt(2 / (m.conv1.weight.shape[0] * np.prod(m.conv1.weight.shape[2:]))) * self.num_layers ** (-0.5))
                nn.init.constant_(m.conv2.weight, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1,algorithm='BP'):
        downsample = None
        if stride != 1:
            downsample = nn.AvgPool2d(1, stride=stride)


        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,algorithm=algorithm))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(planes, planes,algorithm=algorithm))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x + self.bias1)

        x = self.layer1(x)
        x = self.layer2(x)
        latent = self.layer3(x)

        x = self.avgpool(latent)
        x = x.view(x.size(0), -1)
        # x = self.flatten(x)
        x = self.fc(x + self.biasfc)

        return latent, x


def fixup_resnet14(**kwargs):
    """Constructs a Fixup-ResNet-14 model.

    """
    model = FixupResNet(FixupBasicBlock, [2, 2, 2], **kwargs)
    return model

def fixup_resnet20(**kwargs):
    """Constructs a Fixup-ResNet-20 model.

    """
    model = FixupResNet(FixupBasicBlock, [3, 3, 3], **kwargs)
    return model


def fixup_resnet32(**kwargs):
    """Constructs a Fixup-ResNet-32 model.

    """
    model = FixupResNet(FixupBasicBlock, [5, 5, 5], **kwargs)
    return model


def fixup_resnet44(**kwargs):
    """Constructs a Fixup-ResNet-44 model.

    """
    model = FixupResNet(FixupBasicBlock, [7, 7, 7], **kwargs)
    return model


def fixup_resnet56(**kwargs):
    """Constructs a Fixup-ResNet-56 model.

    """
    model = FixupResNet(FixupBasicBlock, [9, 9, 9], **kwargs)
    return model


def fixup_resnet110(**kwargs):
    """Constructs a Fixup-ResNet-110 model.

    """
    model = FixupResNet(FixupBasicBlock, [18, 18, 18], **kwargs)
    return model


def fixup_resnet1202(**kwargs):
    """Constructs a Fixup-ResNet-1202 model.

    """
    model = FixupResNet(FixupBasicBlock, [200, 200, 200], **kwargs)
    return model    


#------------------------

import torch
import torch.nn as nn
import numpy as np
from modules import customized_modules
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
Conv2d = customized_modules.AsymmetricFeedbackConv2d 
ConvTranspose2d = customized_modules.AsymmetricFeedbackConvTranspose2d
Linear = customized_modules.LinearModule

__all__ = ['FixupResNet', 'fixup_resnet20', 'fixup_resnet32', 'fixup_resnet44', 'fixup_resnet56', 'fixup_resnet110', 'fixup_resnet1202']


def convT3x3(in_planes, out_planes, stride=1,algorithm='BP'):
    """3x3 convolution with padding"""
    return ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, algorithm=algorithm)

def convT1x1(in_planes, out_planes, stride=1, algorithm='BP'):
    """1x1 convolution"""
    return ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, algorithm=algorithm)



class FixupBasicBlockT(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None,algorithm='BP'):
        super(FixupBasicBlockT, self).__init__()
        # Both self.conv1 and self.upsample layers upsample the input when stride != 1
        self.bias1a = nn.Parameter(torch.zeros(1))
        self.conv2 = convT3x3(planes, planes,algorithm=algorithm)
        self.bias1b = nn.Parameter(torch.zeros(1))
        self.relu = nn.ReLU(inplace=False)

        self.bias2a = nn.Parameter(torch.zeros(1))
        self.conv1 = convT3x3(planes, inplanes, stride,algorithm=algorithm)
        self.scale = nn.Parameter(torch.ones(1))
        self.bias2b = nn.Parameter(torch.zeros(1))
        self.upsample = upsample

    def forward(self, x):
        identity = x

        out = self.conv2(x + self.bias1a)
        out = self.relu(out + self.bias1b)

        out = self.conv1(out + self.bias2a)
        out = out * self.scale + self.bias2b

        if self.upsample is not None:

            # An us
            x_less_channels = torch.zeros_like(x)
            x_less_channels = x_less_channels[:,:int(x.shape[1]/2),:,:]
            for ib, b in enumerate(x):
                diff = []
                for rf in b:
                    diff.extend([torch.max(rf).item() - torch.min(rf).item()])
                ind_important_rfs = np.argsort(diff)[int(len(diff)/2):]
                x_less_channels[ib] = x[ib,ind_important_rfs]
            
            x = x_less_channels
            
            identity = self.upsample(x + self.bias1a)
            out = nn.functional.pad(out, pad=(1,0,1,0), mode='constant', value=0)
            # identity = torch.cat((identity, torch.zeros_like(identity)), 1)

        # print('out:',out.shape,'identity', identity.shape)
        out += identity
        out = self.relu(out)

        return out


class FixupResNetT(nn.Module):

    def __init__(self, block, layers, num_classes=10,algorithm='BP'):
        super(FixupResNetT, self).__init__()
        self.num_layers = sum(layers)

        self.inplanes = 16*4
        self.conv1 = convT3x3(16, 1, algorithm=algorithm)
        self.bias1 = nn.Parameter(torch.zeros(1))
        self.relu = nn.ReLU(inplace=False)
        self.layer3 = self._make_layer(block, 32, layers[2], stride=2,algorithm=algorithm)
        self.layer2 = self._make_layer(block, 16, layers[1], stride=2,algorithm=algorithm)
        self.layer1 = self._make_layer(block, 16, layers[0], algorithm=algorithm)


        for m in self.modules():
            if isinstance(m, FixupBasicBlockT):
                nn.init.normal_(m.conv2.weight, mean=0, std=np.sqrt(2 / (m.conv1.weight.shape[0] * np.prod(m.conv1.weight.shape[2:]))) * self.num_layers ** (-0.5))
                nn.init.constant_(m.conv1.weight, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1,algorithm='BP'):
        upsample = None

        layers = []
        
        for _ in range(blocks-1):
            layers.append(block(self.inplanes, self.inplanes,algorithm=algorithm))

        if stride != 1 or self.inplanes != int(planes / block.expansion):
            upsample = nn.Upsample(scale_factor=2, mode='bilinear')
            # upsample =  convT1x1(self.inplanes, int(planes / block.expansion),  stride, algorithm=algorithm)

        layers.append(block(planes, self.inplanes, stride, upsample,algorithm=algorithm))
        self.inplanes = planes
        
        return nn.Sequential(*layers)

    def forward(self, x):
        
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)

        x = self.relu(x)
        x = self.conv1(x+ self.bias1)

        return x

def fixup_resnetT14(**kwargs):
    """Constructs a Fixup-ResNet-14 model.

    """
    model = FixupResNetT(FixupBasicBlockT, [2, 2, 2], **kwargs)
    return model


def fixup_resnetT20(**kwargs):
    """Constructs a Fixup-ResNet-20 model.

    """
    model = FixupResNetT(FixupBasicBlockT, [3, 3, 3], **kwargs)
    return model


def fixup_resnetT32(**kwargs):
    """Constructs a Fixup-ResNet-32 model.

    """
    model = FixupResNetT(FixupBasicBlockT, [5, 5, 5], **kwargs)
    return model


def fixup_resnetT44(**kwargs):
    """Constructs a Fixup-ResNet-44 model.

    """
    model = FixupResNetT(FixupBasicBlockT, [7, 7, 7], **kwargs)
    return model


def fixup_resnetT56(**kwargs):
    """Constructs a Fixup-ResNet-56 model.

    """
    model = FixupResNetT(FixupBasicBlockT, [9, 9, 9], **kwargs)
    return model


def fixup_resnetT110(**kwargs):
    """Constructs a Fixup-ResNet-110 model.

    """
    model = FixupResNetT(FixupBasicBlockT, [18, 18, 18], **kwargs)
    return model


def fixup_resnetT1202(**kwargs):
    """Constructs a Fixup-ResNet-1202 model.

    """
    model = FixupResNetT(FixupBasicBlockT, [200, 200, 200], **kwargs)
    return model    