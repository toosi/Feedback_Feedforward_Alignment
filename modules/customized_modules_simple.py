import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import copy
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _single, _pair


def linear_fa_backward_hook(module, grad_input, grad_output):
    if grad_input[1] is not None:
        grad_input_fa = grad_output[0].mm(module.weight_feedback)
        return (grad_input[0], grad_input_fa) + grad_input[2:]

def conv2d_fa_backward_hook(module, grad_input, grad_output):
    if grad_input[0] is not None:
        grad_input_fa = torch.nn.grad.conv2d_input(grad_input[0].size(), module.weight_feedback, grad_output[0], stride=module.stride, padding=module.padding, dilation=module.dilation, groups=module.groups)
        return (grad_input_fa,) + grad_input[1:]

def convtranspose2d_fa_backward_hook(module, grad_input, grad_output):
    # print(module)
    # print(grad_output[0].shape)
    if grad_input[0] is not None:
        grad_input_fa = convTranspose2d_input(grad_input[0].size(), module.weight_feedback, grad_output[0], stride=module.stride, padding=module.padding, dilation=module.dilation, groups=module.groups)
        return (grad_input_fa,) + grad_input[1:]


def convTranspose2d_input(input_size, weight, grad_output, stride=1, padding=0, output_padding=0, dilation=1, groups=1):
    r"""
    TT copied from https://github.com/pytorch/pytorch/blob/master/torch/nn/grad.py
    Computes the gradient of conv2d(trasnposed) with respect to the input of the convolution.
    This is same as the 2D transposed convolution operator under the hood but requires
    the shape of the gradient w.r.t. input to be specified explicitly.
    Args:
        input_size : Shape of the input gradient tensor
        weight: weight tensor (out_channels x in_channels/groups x kH x kW)
        grad_output : output gradient tensor (minibatch x out_channels x oH x oW)
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
    Examples::
        >>> input = torch.randn(1,1,3,3, requires_grad=True)
        >>> weight = torch.randn(1,1,1,2, requires_grad=True)
        >>> output = F.conv2d(input, weight)
        >>> grad_output = torch.randn(output.shape)
        >>> grad_input = torch.autograd.grad(output, input, grad_output)
        >>> F.grad.conv2d_input(input.shape, weight, grad_output)
    """
    stride = _pair(stride)
    padding = _pair(padding)
    output_padding = _pair(output_padding)
    dilation = _pair(dilation)
    kernel_size = (weight.shape[2], weight.shape[3])
    if input_size is None:
        raise ValueError("grad.conv2d_input requires specifying an input_size")
    # grad_input_padding = torch.nn.grad._grad_input_padding(grad_output, input_size, stride,
    #                                          padding, kernel_size)
    return torch.conv2d(
        grad_output, weight, None, stride, padding, dilation, groups)

# conv2d ---> n_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'

# conv_transpose2d ---> n_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros'

# def convtranspose2d_fa_backward_hook(module, grad_input, grad_output):
#     if grad_input[0] is not None:
#         grad_input_fa = convTranspose2d_input(grad_input[0].size(), module.weight_feedback, grad_output[0], stride=module.stride, padding=module.padding, output_padding=module.output_padding, dilation=module.dilation, groups=module.groups)
#         return (grad_input_fa,) + grad_input[1:]

class LinearModule(nn.Module):
    """
    Implementation of a linear module which uses random feedback weights
    in its backward pass, as described in Lillicrap et al., 2016:
    https://www.nature.com/articles/ncomms13276
    """
    __constants__ = ['bias', 'in_features', 'out_features']
    def __init__(self, in_features, out_features, bias=False, algorithm='FA'):
        super(LinearModule, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.algorithm = algorithm

        if self.algorithm == 'BP':
            self.weight_feedback = copy.deepcopy(self.weight) #.clone()
        else:
            self.weight_feedback = Parameter(torch.FloatTensor(out_features, in_features), requires_grad=False)
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.register_backward_hook(linear_fa_backward_hook)

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.algorithm == 'FA':
            init.kaiming_uniform_(self.weight_feedback, a=math.sqrt(5))
        elif self.algorithm == 'BP':
            self.weight_feedback = copy.deepcopy(self.weight)

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
    def forward(self, input):
        return F.linear(input, self.weight, self.bias)
    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class _ConvNdFA(nn.Module):
    """
    Implementation of an N-dimensional convolution module which uses random feedback weights
    in its backward pass, as described in Lillicrap et al., 2016:
    https://www.nature.com/articles/ncomms13276
    This code is exactly copied from the _ConvNd module in PyTorch, with the addition
    of the random feedback weights.
    """
    __constants__ = ['stride', 'padding', 'dilation', 'groups', 'bias', 'padding_mode']
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding,
                 groups, padding_mode, bias=False, algorithm='FA'):
        super(_ConvNdFA, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode
        self.algorithm = algorithm
        if transposed:
            self.weight = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))

            self.weight_feedback = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size), requires_grad=False)
        else:
            self.weight = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))

            self.weight_feedback = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size), requires_grad=False)
        
        if self.algorithm == 'BP':
            self.weight_feedback = copy.deepcopy(self.weight)
            
            
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.algorithm == 'FA':
            init.kaiming_uniform_(self.weight_feedback, a=math.sqrt(5))
        elif self.algorithm == 'BP':
            self.weight_feedback = copy.deepcopy(self.weight)

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)


class AsymmetricFeedbackConv2d(_ConvNdFA):
    """
    Implementation of a 2D convolution module which uses random feedback weights
    in its backward pass, as described in Lillicrap et al., 2016:
    https://www.nature.com/articles/ncomms13276
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=False, padding_mode='zeros', algorithm='FA'):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        self.algorithm = algorithm
        super(AsymmetricFeedbackConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            transposed=False, output_padding=_pair(0), groups=groups, bias=bias, padding_mode=padding_mode, algorithm=algorithm)
        if self.algorithm == 'FA':
            self.register_backward_hook(conv2d_fa_backward_hook)
    def forward(self, input):
         
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                            self.weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


# form https://pytorch.org/docs/stable/_modules/torch/nn/modules/conv.html#ConvTranspose2d
class _ConvTransposeMixin(object):
    def _output_padding(self, input, output_size, stride, padding, kernel_size):
        # type: (Tensor, Optional[List[int]], List[int], List[int], List[int]) -> List[int]
        if output_size is None:
            ret = _single(self.output_padding)  # converting to list if was not already
        else:
            k = input.dim() - 2
            if len(output_size) == k + 2:
                output_size = output_size[2:]
            if len(output_size) != k:
                raise ValueError(
                    "output_size must have {} or {} elements (got {})"
                    .format(k, k + 2, len(output_size)))
            min_sizes = torch.jit.annotate(List[int], [])
            max_sizes = torch.jit.annotate(List[int], [])
            for d in range(k):
                dim_size = ((input.size(d + 2) - 1) * stride[d] -
                            2 * padding[d] + kernel_size[d])
                min_sizes.append(dim_size)
                max_sizes.append(min_sizes[d] + stride[d] - 1)
            for i in range(len(output_size)):
                size = output_size[i]
                min_size = min_sizes[i]
                max_size = max_sizes[i]
                if size < min_size or size > max_size:
                    raise ValueError((
                        "requested an output size of {}, but valid sizes range "
                        "from {} to {} (for an input of {})").format(
                            output_size, min_sizes, max_sizes, input.size()[2:]))
            res = torch.jit.annotate(List[int], [])
            for d in range(k):
                res.append(output_size[d] - min_sizes[d])
            ret = res
        return ret
        
class AsymmetricFeedbackConvTranspose2d(_ConvTransposeMixin, _ConvNdFA):
    r"""Applies a 2D transposed convolution operator over an input image
    composed of several input planes.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=False,
                 dilation=1, padding_mode='zeros', algorithm='FA'):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        output_padding = _pair(output_padding)
        super(AsymmetricFeedbackConvTranspose2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            transposed=True, output_padding=output_padding, groups=groups, bias=bias, padding_mode=padding_mode, algorithm=algorithm)
        if algorithm == 'FA':
            self.register_backward_hook(convtranspose2d_fa_backward_hook)

    def forward(self, input, output_size=None):
        # type: (Tensor, Optional[List[int]]) -> Tensor
        # if self.padding_mode != 'zeros':
        #     raise ValueError('Only `zeros` padding mode is supported for ConvTranspose2d, %s is not supported'%self.padding_mode)
        output_padding = self._output_padding(input, output_size, self.stride, self.padding, self.kernel_size)
        return F.conv_transpose2d(
            input, self.weight, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)