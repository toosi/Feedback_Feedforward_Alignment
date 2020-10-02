import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn import init
import torch.autograd as autograd

import warnings
from torch.nn.modules.utils import _single, _pair
import math
import copy

class ReLUFunction(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input

class ReLU(nn.Module):
    def __init__(self):
        super(ReLU, self).__init__()
    
    def forward(self, input, input2):
        return ReLUFunction.apply(input)


class LinearFunction(autograd.Function):

    """
    Autograd function for a linear layer with asymmetric feedback and feedforward pathways
    forward  : weight
    backward : weight_feedback
    bias is set to None for now
    """

    @staticmethod
    # same as reference linear function, but with additional fa tensor for backward
    def forward(context, input, weight, weight_feedback, bias=None, algorithm_id=0):
        context.save_for_backward(input, weight, weight_feedback, bias, algorithm_id)
        output = input.mm(weight.t())

        if bias is not None:
            output  += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(context, grad_output):
        input, weight, weight_feedback, bias, algorithm_id = context.saved_tensors
        grad_input = grad_weight = grad_weight_feedback = grad_bias = None

        if context.needs_input_grad[0]:
            # all of the logic of FA resides in this one line
            # calculate the gradient of input with fixed fa tensor, rather than the "correct" model weight
            grad_input = grad_output.mm(weight_feedback)
        if context.needs_input_grad[1]:
            # grad for weight with FA'ed grad_output from downstream layer
            # it is same with original linear function
            grad_weight = grad_output.t().mm(input) # (sorta)Hebbian update for forward
        
        if context.needs_input_grad[2]:
            # only SL needs gradients for backward weights

            grad_weight_feedback = grad_output.t().mm(input)  # (sorta)Hebbian update for backward
            
        if bias is not None and context.needs_input_grad[3]:
            grad_bias = grad_output.sum(0).squeeze(0)
        
        grad_weight_feedback = None

        return grad_input, grad_weight, grad_weight_feedback, grad_bias, None

class Linear(nn.Module):

    """
    a linear layer with asymmetric feedback and feedforward pathways
    forward  : weight
    backward : weight_feedback
    """

    def __init__(self, input_features, output_features, bias, algorithm ):     # we ignore bias for now
        
        # implemented_algorithms = ['BP', 'FA', 'SL']
        # assert algorithm in implemented_algorithms, 'feedback algorithm %s is not implemented'

        super(Linear, self).__init__()
        # self.input_features = input_features
        # self.output_features = output_features
        self.algorithm = algorithm
        # weight and bias for forward pass
        # weight has transposed form for efficiency (?) (transposed at forward pass)

        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        # as in torchvision/nn/modules/linear scaling was based on weight input (weight.size(1))
        # since  weight_feedback is the transpose scaling should be like below
#         self.scale_feedback = 1. / math.sqrt(self.weight.size(0))
        if bias:  
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else: 
            self.register_parameter('bias', None)
        if self.algorithm == 'SL':
            back_requires_grad = True
        else:
            back_requires_grad = False
    
        self.weight_feedback = nn.Parameter(torch.Tensor(output_features, input_features), 
                                            requires_grad=back_requires_grad)

        self.reset_parameters()
        if self.algorithm == 'BP':
            self.weight_feedback.data = copy.deepcopy(self.weight.detach())


        
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_feedback, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):

        # if self.algorithm == 'FA':

        #     weight_feedback = self.weight_feedback
        if self.algorithm == 'BP':

            self.weight_feedback.data = copy.deepcopy(self.weight.detach())


        return LinearFunction.apply(input, self.weight, self.weight_feedback, self.bias, self.algorithm_id)
        

# ------ Convolution

def _grad_input_padding(grad_output, input_size, stride, padding, kernel_size, dilation=None):
    if dilation is None:
        # For backward compatibility
        warnings.warn("_grad_input_padding 'dilation' argument not provided. Default of 1 is used.")
        dilation = [1] * len(stride)

    input_size = list(input_size)
    k = grad_output.dim() - 2

    if len(input_size) == k + 2:
        input_size = input_size[-k:]
    if len(input_size) != k:
        raise ValueError("input_size must have {} elements (got {})"
                         .format(k + 2, len(input_size)))

    def dim_size(d):
        return ((grad_output.size(d + 2) - 1) * stride[d] - 2 * padding[d] + 1
                + dilation[d] * (kernel_size[d] - 1))

    min_sizes = [dim_size(d) for d in range(k)]
    max_sizes = [min_sizes[d] + stride[d] - 1 for d in range(k)]
    for size, min_size, max_size in zip(input_size, min_sizes, max_sizes):
        if size < min_size or size > max_size:
            raise ValueError(
                ("requested an input grad size of {}, but valid sizes range "
                 "from {} to {} (for a grad_output of {})").format(
                     input_size, min_sizes, max_sizes,
                     grad_output.size()[2:]))

    return tuple(input_size[d] - min_sizes[d] for d in range(k))

def conv2d_input(input_size, weight, grad_output, stride=1, padding=0, dilation=1, groups=1):
    r"""
    Computes the gradient of conv2d with respect to the input of the convolution.
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
    dilation = _pair(dilation)
    kernel_size = (weight.shape[2], weight.shape[3])

    if input_size is None:
        raise ValueError("grad.conv2d_input requires specifying an input_size")

    grad_input_padding = _grad_input_padding(grad_output, input_size, stride,
                                             padding, kernel_size, dilation)

    return torch.conv_transpose2d(
        grad_output, weight, None, stride, padding, grad_input_padding, groups,
        dilation)


def conv2d_weight(input, weight_size, grad_output, stride=1, padding=0, dilation=1, groups=1):
    r"""
    Computes the gradient of conv2d with respect to the weight of the convolution.
    Args:
        input: input tensor of shape (minibatch x in_channels x iH x iW)
        weight_size : Shape of the weight gradient tensor
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
        >>> grad_weight = torch.autograd.grad(output, filter, grad_output)
        >>> F.grad.conv2d_weight(input, weight.shape, grad_output)
    """
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)
    in_channels = input.shape[1]
    out_channels = grad_output.shape[1]
    min_batch = input.shape[0]

    grad_output = grad_output.contiguous().repeat(1, in_channels // groups, 1,
                                                  1)
    grad_output = grad_output.contiguous().view(
        grad_output.shape[0] * grad_output.shape[1], 1, grad_output.shape[2],
        grad_output.shape[3])

    input = input.contiguous().view(1, input.shape[0] * input.shape[1],
                                    input.shape[2], input.shape[3])

    grad_weight = torch.conv2d(input, grad_output, None, dilation, padding,
                               stride, in_channels * min_batch)

    grad_weight = grad_weight.contiguous().view(
        min_batch, grad_weight.shape[1] // min_batch, grad_weight.shape[2],
        grad_weight.shape[3])

    return grad_weight.sum(dim=0).view(
        in_channels // groups, out_channels,
        grad_weight.shape[2], grad_weight.shape[3]).transpose(0, 1).narrow(
            2, 0, weight_size[2]).narrow(3, 0, weight_size[3])


class Conv2dFunction(autograd.Function):

    """
    Autograd function for a linear layer with asymmetric feedback and feedforward pathways
    forward  : weight
    backward : weight_feedback
    bias is set to None for now
    """

    @staticmethod
    # same as reference linear function, but with additional fa tensor for backward
    def forward(context, input, weight, weight_feedback, bias,
                stride, padding, dilation, groups):
        context.save_for_backward(input, weight, weight_feedback, bias,
                                  stride, padding, dilation, groups)

#         if self.padding_mode == 'circular':
#                 expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
#                                     (self.padding[0] + 1) // 2, self.padding[0] // 2)
#                 return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
#                                 self.weight, self.bias, self.stride,
#                                 _pair(0), self.dilation, self.groups)
        stride = _pair(stride.item())
        padding = _pair(padding.item())
        dilation = _pair(dilation.item())
        return F.conv2d(input, weight, bias, stride,
                        padding, dilation, groups)

    @staticmethod
    def backward(context, grad_output):
        input, weight, weight_feedback, bias, stride, padding, dilation, groups = context.saved_tensors
        grad_input = grad_weight = grad_weight_feedback = grad_bias = None
        
        
        stride = _pair(stride.item())
        padding = _pair(padding.item())
        dilation = _pair(dilation.item())
        
        input_size = input.shape
        
        if context.needs_input_grad[0]:
            # asymmetric path happens here
            grad_input = conv2d_input(input_size, weight_feedback, grad_output, stride, 
                                      padding, dilation, groups)
            
        if context.needs_input_grad[1]:
            # hebbian forward
            weight_size = weight.shape
            grad_weight = conv2d_weight(input, weight_size, grad_output, stride, 
                                        padding, dilation, groups) # (sorta)Hebbian update for backward
        
        if context.needs_input_grad[2]:
            # hebbian backward
            # only SL needs gradients for backward weights

            weight_feedback_size = weight.shape
            grad_weight_feedback = conv2d_weight(input, weight_feedback_size, grad_output, stride , 
                                        padding, dilation, groups) # (sorta)Hebbian update for backward
            
        if bias is not None and context.needs_input_grad[3]:
            grad_bias = grad_output.sum(0).squeeze(0)
        
        grad_weight_feedback = None
        return grad_input, grad_weight, grad_weight_feedback, grad_bias, None, None, None, None





class Conv2d(nn.Module):
    """
    Implementation of an N-dimensional convolution module which uses random feedback weights
    in its backward pass, as described in Lillicrap et al., 2016:
    https://www.nature.com/articles/ncomms13276
    This code is exactly copied from the _ConvNd module in PyTorch, with the addition
    of the random feedback weights.
    """

    __constants__ = ['stride', 'padding', 'dilation', 'groups', 'bias', 'padding_mode']

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', algorithm='BP'):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2d, self).__init__()
        
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
            
        self.algorithm = algorithm
        # implemented_algorithms = ['BP', 'FA', 'SL']
        # self.algorithm_id = nn.Parameter(torch.tensor(implemented_algorithms.index(algorithm)), requires_grad=False)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = 0 #transposed
        self.output_padding = (0,0) # output_padding
        self.groups = groups
        self.padding_mode = padding_mode
        
        if self.algorithm == 'SL':
            back_requires_grad = True
        else:
            back_requires_grad = False
        
        # I keep the transposed here in case I decide to implement a customized ConvTranspose2d
            
        if self.transposed:
            self.weight = nn.Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))

            self.weight_feedback = nn.Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size), requires_grad=back_requires_grad)
        else:
            self.weight = nn.Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))

            self.weight_feedback = nn.Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size), requires_grad=back_requires_grad)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_feedback, a=math.sqrt(5))
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
    
    def forward(self, input):
        if self.algorithm == 'BP':

            self.weight_feedback.data = copy.deepcopy(self.weight.detach()) 
        
        return Conv2dFunction.apply(input, self.weight, self.weight_feedback, self.bias,
                                    Variable(torch.tensor(self.stride[0])), 
                                    Variable(torch.tensor(self.padding[0])), 
                                    Variable(torch.tensor(self.dilation[0])), 
                                    Variable(torch.tensor(self.groups))
                                    )


    

from torch.autograd import Function
from torch.autograd.function import once_differentiable
import torch.nn.functional as F

def unsqueeze(tensor):
    return tensor.unsqueeze(1).unsqueeze(0)
    
class SyncBNFunction(Function):

    @staticmethod
    def forward(ctx, x, weight, bias, weight_feedback, running_mean, running_var, momentum, eps, training, bn_ctx):
        x_shape = x.shape
        B, C = x_shape[:2]

        _x = x.view(B,C,-1).contiguous()

        ctx.eps = eps
        ctx.training = training

        ctx.sync = bn_ctx.sync
        ctx.cur_device = bn_ctx.cur_device
        ctx.queue = bn_ctx.queue
        ctx.is_master = bn_ctx.is_master
        ctx.devices = bn_ctx.devices

        norm = 1/(_x.shape[0] * _x.shape[2])

        if ctx.training:
            _ex = _x.sum(2).sum(0) * norm
            _exs = _x.pow(2).sum(2).sum(0) * norm

            if ctx.sync:
                if ctx.is_master:

                    _ex, _exs = [_ex.unsqueeze(1)], [_exs.unsqueeze(1)]

                    master_queue = ctx.queue[0]
                    for j in range(master_queue.maxsize):
                        _slave_ex, _slave_exs = master_queue.get()
                        master_queue.task_done()

                        _ex.append(  _slave_ex.unsqueeze(1)  )
                        _exs.append( _slave_exs.unsqueeze(1) )
                    
                    _ex  = torch.cuda.comm.gather( _ex,  dim=1 ).mean(1)
                    _exs = torch.cuda.comm.gather( _exs, dim=1 ).mean(1)

                    distributed_tensor = torch.cuda.comm.broadcast_coalesced( (_ex, _exs), ctx.devices )

                    for dt, q in zip( distributed_tensor[1:], ctx.queue[1:] ):
                        q.put(dt)
                else:
                    master_queue = ctx.queue[0]
                    slave_queue = ctx.queue[ctx.cur_device]
                    master_queue.put( (_ex, _exs) )

                    _ex, _exs = slave_queue.get()
                    slave_queue.task_done()
                    _ex, _exs = _ex.squeeze(), _exs.squeeze()
            
            _var = _exs - _ex.pow(2)
            N = B*len(ctx.devices)
            unbiased_var = _var * N / (N - 1)
            
            running_mean.mul_( (1-momentum) ).add_( momentum * _ex  )
            running_var.mul_( (1-momentum) ).add_( momentum * unbiased_var )
            # ctx.mark_dirty(running_mean, running_var)
        else:
            _ex, _var = running_mean.contiguous(), running_var.contiguous()
            _exs = _ex.pow(2) + _var
            
        invstd = 1/torch.sqrt( _var + eps )

        if weight is not None: # affine
            output = (_x - unsqueeze(_ex) ) * unsqueeze(invstd) * unsqueeze(weight)  + unsqueeze(bias)
        else:
            output = (_x - unsqueeze(_ex) ) * unsqueeze(invstd)
        
        ctx.save_for_backward(x, _ex, _exs, weight, bias,weight_feedback)
        return output.view(*x_shape).contiguous().clone()

    @staticmethod
    def backward(ctx, grad_output):
        x, _ex, _exs, weight, bias, weight_feedback = ctx.saved_tensors
        grad_x = grad_weight = grad_bias = grad_weight_feedback = None

        B,C = grad_output.shape[:2]
        grad_output_shape = grad_output.shape

        _var = _exs - _ex.pow(2)
        _std = torch.sqrt( _var + ctx.eps)
        invstd = 1.0 / _std

        grad_output = grad_output.view(B,C,-1)
        x = x.view(B,C,-1)

        norm = 1.0/(x.shape[0] * x.shape[2])
        
        dot_p = ( grad_output * ( x -  unsqueeze( _ex ) ) ).sum(2).sum(0)
        grad_output_sum = grad_output.sum(2).sum(0)

        grad_scale = weight_feedback * invstd #weight * invstd

        grad_ex  = -grad_output_sum * grad_scale + _ex * invstd * invstd * dot_p * grad_scale
        grad_exs = -0.5 * grad_scale * invstd * invstd * dot_p 

        # Sync
        if ctx.training:
            if ctx.sync:
                if ctx.is_master: 
                    grad_ex, grad_exs = [grad_ex.unsqueeze(1)], [grad_exs.unsqueeze(1)]
                    master_queue = ctx.queue[0]
                    for j in range(master_queue.maxsize):
                        grad_slave_ex, grad_slave_exs = master_queue.get()
                        master_queue.task_done()

                        grad_ex.append(  grad_slave_ex.unsqueeze(1)  )
                        grad_exs.append( grad_slave_exs.unsqueeze(1) )

                    grad_ex  = torch.cuda.comm.gather( grad_ex,  dim=1 ).mean(1)
                    grad_exs = torch.cuda.comm.gather( grad_exs, dim=1).mean(1)

                    distributed_tensor = torch.cuda.comm.broadcast_coalesced( (grad_ex, grad_exs), ctx.devices )
                    for dt, q in zip( distributed_tensor[1:], ctx.queue[1:] ):
                        q.put(dt)
                else:
                    master_queue = ctx.queue[0]
                    slave_queue = ctx.queue[ctx.cur_device]
                    master_queue.put( (grad_ex, grad_exs) )

                    grad_ex, grad_exs = slave_queue.get()
                    slave_queue.task_done()
                    grad_ex, grad_exs = grad_ex.squeeze(), grad_exs.squeeze()
        
        if ctx.needs_input_grad[0]:
            grad_x = grad_output * unsqueeze( grad_scale ) + unsqueeze( grad_ex * norm ) +  unsqueeze(grad_exs) * 2 * x * norm 

        if ctx.needs_input_grad[1]:
            grad_weight = dot_p * invstd
        
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output_sum

        if ctx.needs_input_grad[3]:
            grad_weight_feedback = dot_p * invstd
        
        return grad_x.view(*grad_output_shape), grad_weight, grad_bias, grad_weight_feedback, None, None, None, None, None, None

        

        


    
# from torch.nn.modules.batchnorm import _BatchNorm
import torch.nn as nn
import torch
import torch.nn.functional as F
from queue import Queue
from torch import Tensor

import collections

_bn_context = collections.namedtuple("_bn_context", ['sync', 'is_master', 'cur_device', 'queue', 'devices'])


class _NormBase(nn.Module):
    """Common base of _InstanceNorm and _BatchNorm"""
    _version = 2
    __constants__ = ['track_running_stats', 'momentum', 'eps',
                     'num_features', 'affine']
    num_features: int
    eps: float
    momentum: float
    affine: bool
    track_running_stats: bool
    # WARNING: weight and bias purposely not defined here.
    # See https://github.com/pytorch/pytorch/issues/39670

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        algorithm: str = 'BP',
    ) -> None:
        super(_NormBase, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        self.algorithm = algorithm
        if self.algorithm == 'SL':
            back_requires_grad = True
        else:
            back_requires_grad = False

        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.weight_feedback = Parameter(torch.Tensor(num_features), requires_grad=back_requires_grad)
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('weight_feedback', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self) -> None:
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self) -> None:
        self.reset_running_stats()
        if self.affine:
            init.ones_(self.weight)
            init.ones_(self.weight_feedback)
            init.zeros_(self.bias)

    def _check_input_dim(self, input):
        raise NotImplementedError

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)

        if (version is None or version < 2) and self.track_running_stats:
            # at version 2: added num_batches_tracked buffer
            #               this should have a default value of 0
            num_batches_tracked_key = prefix + 'num_batches_tracked'
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)

        super(_NormBase, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)


class _BatchNorm(_NormBase):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, algorithm='BP'):
        super(_BatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats, algorithm)

    def forward(self, input: Tensor) -> Tensor:
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r""" 
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        return F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight, self.bias, bn_training, exponential_average_factor, self.eps)


class SyncBatchNorm(_BatchNorm):
    """ Sync BN
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, algorithm='BP'):
        super(SyncBatchNorm, self).__init__(num_features, eps=eps, momentum=momentum, affine=affine, 
        track_running_stats=track_running_stats, algorithm='BP')

        self.devices = list(range(torch.cuda.device_count()))
        self.sync = len(self.devices)>1
        self._slaves = self.devices[1:]
        self._queues = [ Queue(len(self._slaves)) ] + [ Queue(1) for _ in self._slaves ]


        self.algorithm = algorithm
        implemented_algorithms = ['BP', 'FA', 'SL']
        self.algorithm_id = nn.Parameter(torch.tensor(implemented_algorithms.index(algorithm)), requires_grad=False)
        
        if self.algorithm == 'SL':
            back_requires_grad = True
        else:
            back_requires_grad = False

    def _check_input_dim(self, input):
        if input.dim() < 2:
            raise ValueError('expected at least 2 dims (got {}D input)'
                             .format(input.dim()))

    def forward(self, input):
        self._check_input_dim(input)
        
        if not self.training and self.track_running_stats:
            return F.batch_norm(input, running_mean=self.running_mean, running_var=self.running_var,
                         weight=self.weight, bias=self.bias, training=False, momentum=0.0, eps=self.eps)
        else:
            exponential_average_factor = 0.0
            if self.num_batches_tracked is not None: # track running statistics
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum
            
            if input.is_cuda:
                cur_device = input.get_device()
                bn_ctx = _bn_context( self.sync, (cur_device==self.devices[0]), cur_device, self._queues, self.devices )
            else:
                bn_ctx = _bn_context( False, True, None, None, None  )
        
        if self.algorithm == 'BP':

            self.weight_feedback.data = copy.deepcopy(self.weight.detach()) 
            
        return SyncBNFunction.apply( input, self.weight, self.bias, self.weight_feedback, self.running_mean, self.running_var, exponential_average_factor, self.eps, self.training, bn_ctx )
    

class SyncBatchNorm1d(SyncBatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))


class SyncBatchNorm2d(SyncBatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))


class SyncBatchNorm3d(SyncBatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))

    


# class BatchNorm2d(nn.BatchNorm2d):
#     def __init__(self, num_features, eps=1e-5, momentum=0.1,
#                  affine=True, track_running_stats=True):
#         super(BatchNorm2d, self).__init__(
#             num_features, eps, momentum, affine, track_running_stats)

#     def forward(self, input):
#         self._check_input_dim(input)

#         exponential_average_factor = 0.0

#         if self.training and self.track_running_stats:
#             if self.num_batches_tracked is not None:
#                 self.num_batches_tracked += 1
#                 if self.momentum is None:  # use cumulative moving average
#                     exponential_average_factor = 1.0 / float(self.num_batches_tracked)
#                 else:  # use exponential moving average
#                     exponential_average_factor = self.momentum

#         # calculate running estimates
#         if self.training:
#             mean = input.mean([0, 2, 3])
#             # use biased var in train
#             var = input.var([0, 2, 3], unbiased=False)
#             n = input.numel() / input.size(1)
#             with torch.no_grad():
#                 self.running_mean = exponential_average_factor * mean\
#                     + (1 - exponential_average_factor) * self.running_mean
#                 # update running_var with unbiased var
#                 self.running_var = exponential_average_factor * var * n / (n - 1)\
#                     + (1 - exponential_average_factor) * self.running_var
#         else:
#             mean = self.running_mean
#             var = self.running_var
            
#         if self.training:
#             bn_training = True
#         else:
#             bn_training = (self.running_mean is None) and (self.running_var is None)

        
#         kwargs = self.training, bn_training, exponential_average_factor,self.track_running_stats 
#         return BatchNorm2dFunction.apply(input,self.weight, self.bias, self.running_mean, self.running_var ,  self.eps, kwargs)

# def Hadamard(one, two):
#     """
#     @author: hughperkins
#     """
# #     if one.size() != two.size():
# #         raise Exception('size mismatch %s vs %s' % (str(list(one.size())), str(list(two.size()))))
#     one.view_as(two)
#     res = one * two
#     assert res.numel() == one.numel()
#     return res
    
# def mulb(T, v):
#     """
#     T: B x C x H x W
#     v: B
#     it is supposed to do Hadamard without broadcasting 
#     @author: TT
#     """
#     tmp = T.permute(1,0,2,3)
#     tmp = tmp.reshape(T.shape[1],T.shape[0]*T.shape[2]*T.shape[3])


#     tmpmul = tmp*v[:,None].expand_as(tmp)
#     return tmpmul.reshape(T.shape)
        
    
# class BatchNorm2dFunction(autograd.Function):

#     """
#     Autograd function for a linear layer with asymmetric feedback and feedforward pathways
#     forward  : weight
#     backward : weight_feedback
#     bias is set to None for now
#     """

#     @staticmethod
#     # same as reference linear function, but with additional fa tensor for backward
#     def forward(context, input, weight, bias, running_mean,running_var,  eps, kwargs):
        
#         training, bn_training, exponential_average_factor,track_running_stats = kwargs
        
# #         print(input.shape, running_mean.shape, running_var.shape)
#         input_hat = (input - running_mean[None, :, None, None])/torch.sqrt(running_var[None, :, None, None] + eps)
#         input_hat.requires_grad = False
#         context.save_for_backward(input,weight, bias, input_hat, running_mean,running_var, Variable(torch.tensor(eps)))
        
        
        
#         return F.batch_norm(
#             input,
#             # If buffers are not to be tracked, ensure that they won't be updated
#             running_mean if not training or track_running_stats else None,
#             running_var if not training or track_running_stats else None,
#             weight, bias, bn_training, exponential_average_factor, eps)

#     @staticmethod
#     def backward(context, grad_output):
#         input,  weight, bias, input_hat, running_mean, running_var, eps = context.saved_tensors
#         eps = eps.item()
#         N = input.shape[0]
#         # grad_input =grad_input_hat = grad_eps = grad_weight = grad_bias = None
#         # grad_running_mean = grad_running_var = None
        
                   
#         #if context.needs_input_grad[5]: # weight : gamma
# #         print(grad_output.shape, input_hat.shape)
#         grad_weight = Hadamard(grad_output,input_hat)#.sum(0).squeeze(0)

#         #if context.needs_input_grad[6]: # bias : beta
#         grad_bias = grad_output.sum(0).squeeze(0)

# #         if context.needs_input_grad[1]: # input_hat
#         grad_input_hat = mulb(grad_output, weight) #tmpmul.reshape(grad_output.shape)
        
# #         if context.needs_input_grad[3]: # running_mean : mu
#         coef_m = (-1/(torch.sqrt(running_var + eps)))
#         grad_running_mean = mulb(grad_input_hat, coef_m).sum(0).squeeze(0) #coef_m * grad_input_hat.sum(0).squeeze(0)
        

#         #if context.needs_input_grad[4]: # running_var : sigma
#         coef = -0.5*(running_var + eps)**(-1.5)

#         grad_running_var = mulb(grad_input_hat*(input-running_mean[None, :, None, None]), coef ).sum(0).squeeze(0)

#         #if context.needs_input_grad[0]: # input
#         coef_inp = Hadamard((1/N)*weight,(running_var + eps)**(-0.5))
#         part1 = -Hadamard(grad_weight, input)
#         part2 = N*grad_output
#         #print(part1.shape, grad_bias.shape)
#         part3 = -grad_bias.expand_as(grad_output)
#         grad_input =  part1 + part2 + part3
                    
            
#         return grad_input, grad_weight.sum(0).squeeze(), grad_bias.squeeze(), None, None, None, None

    




# class BatchNorm2d(nn.BatchNorm2d):
#     def __init__(self, num_features, eps=1e-5, momentum=0.1,
#                  affine=True, track_running_stats=True):
#         super(BatchNorm2d, self).__init__(
#             num_features, eps, momentum, affine, track_running_stats)
#         self.num_features = num_features
#         self.eps = eps
#         self.momentum = momentum
#         self.affine = affine
#         self.track_running_stats = track_running_stats
#         if self.affine:
#             self.weight = Parameter(torch.Tensor(num_features))
#             self.bias = Parameter(torch.Tensor(num_features))
#         else:
#             self.register_parameter('weight', None)
#             self.register_parameter('bias', None)
#         if self.track_running_stats:
#             self.register_buffer('running_mean', torch.zeros(num_features))
#             self.register_buffer('running_var', torch.ones(num_features))
#             self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
#         else:
#             self.register_parameter('running_mean', None)
#             self.register_parameter('running_var', None)
#             self.register_parameter('num_batches_tracked', None)
#         self.reset_parameters()

#     def reset_running_stats(self) -> None:
#         if self.track_running_stats:
#             self.running_mean.zero_()
#             self.running_var.fill_(1)
#             self.num_batches_tracked.zero_()

#     def reset_parameters(self) -> None:
#         self.reset_running_stats()
#         if self.affine:
#             init.ones_(self.weight)
#             init.zeros_(self.bias)

# #     def _check_input_dim(self, input):
# #         raise NotImplementedError

#     def extra_repr(self):
#         return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
#                'track_running_stats={track_running_stats}'.format(**self.__dict__)



#     def forward(self, input):
#         self._check_input_dim(input)

#         exponential_average_factor = 0.0

#         if self.training and self.track_running_stats:
#             if self.num_batches_tracked is not None:
#                 self.num_batches_tracked += 1
#                 if self.momentum is None:  # use cumulative moving average
#                     exponential_average_factor = 1.0 / float(self.num_batches_tracked)
#                 else:  # use exponential moving average
#                     exponential_average_factor = self.momentum

#         # calculate running estimates
#         if self.training:
#             mean = input.mean([0, 2, 3])
#             # use biased var in train
#             var = input.var([0, 2, 3], unbiased=False)
#             n = input.numel() / input.size(1)
#             with torch.no_grad():
#                 self.running_mean = exponential_average_factor * mean\
#                     + (1 - exponential_average_factor) * self.running_mean
#                 # update running_var with unbiased var
#                 self.running_var = exponential_average_factor * var * n / (n - 1)\
#                     + (1 - exponential_average_factor) * self.running_var
#         else:
#             mean = self.running_mean
#             var = self.running_var
            
#         if self.training:
#             bn_training = True
#         else:
#             bn_training = (self.running_mean is None) and (self.running_var is None)

        
#         kwargs = self.training, bn_training, exponential_average_factor,self.track_running_stats 
#         return BatchNorm2dFunction.apply(input,self.weight, self.bias, self.running_mean, self.running_var ,  self.eps, kwargs)

# def Hadamard(one, two):
#     """
#     @author: hughperkins
#     """
#     # if one.size() != two.size():
#     #     raise Exception('size mismatch %s vs %s' % (str(list(one.size())), str(list(two.size()))))
#     # print('one:',one.shape, 'two', two.shape)
#     try:
#         one.view_as(two)
#     except:
#         if len(two.shape) ==1:
    
#             two = two[None, :, None, None]
#         two.expand_as(one)
    
#     res = one * two
#     assert res.numel() == one.numel()
#     return res
    
# def mulb(T, v):
#     """
#     T: B x C x H x W
#     v: B
#     it is supposed to do Hadamard without broadcasting 
#     @author: TT
#     """
#     tmp = T.permute(1,0,2,3)
#     tmp = tmp.reshape(T.shape[1],T.shape[0]*T.shape[2]*T.shape[3])


#     tmpmul = tmp*v[:,None].expand_as(tmp)
#     return tmpmul.reshape(T.shape)
        
    

# class BatchNorm2dFunction(autograd.Function):

#     """
#     Autograd function for a linear layer with asymmetric feedback and feedforward pathways
#     forward  : weight
#     backward : weight_feedback
#     bias is set to None for now
#     """

#     @staticmethod
#     # same as reference linear function, but with additional fa tensor for backward
#     def forward(context, input, weight, bias, running_mean, running_var, eps, kwargs):
        
#         training, bn_training, exponential_average_factor,track_running_stats = kwargs
        
# #         print(input.shape, running_mean.shape, running_var.shape)
#         input_hat = (input - running_mean[None, :, None, None])/torch.sqrt(running_var[None, :, None, None] + eps)
#         input_hat.requires_grad = False
#         context.save_for_backward(input,weight, bias, input_hat, running_mean, running_var, Variable(torch.tensor(eps)))
        
        
        
#         return F.batch_norm(
#             input,
#             # If buffers are not to be tracked, ensure that they won't be updated
#             running_mean if not training or track_running_stats else None,
#             running_var if not training or track_running_stats else None,
#             weight, bias, bn_training, exponential_average_factor, eps)

#     @staticmethod
#     def backward(context, grad_output):
#         input,  weight, bias, input_hat, running_mean, running_var, eps = context.saved_tensors
#         eps = eps.item()
#         N = input.shape[0]
#         # grad_input =grad_input_hat = grad_eps = grad_weight = grad_bias = None
#         # grad_running_mean = grad_running_var = None
        
                   
#         #if context.needs_input_grad[5]: # weight : gamma
#         # print('grad_output.shape',grad_output.shape, 'input_hat.shape',input_hat.shape)
        
#         grad_weight = torch.einsum('bijk,bijk->ijk', input_hat, grad_output)#.squeeze()#Hadamard(grad_output, input_hat)
#         # print('grad_weight.shape', grad_weight.shape)
#         #if context.needs_input_grad[6]: # bias : beta
#         grad_bias = torch.einsum('bijk,bijk->ijk', torch.ones_like(input_hat), grad_output)#.squeeze()
#         # print('grad_bias.shape', grad_bias.shape)
# #         if context.needs_input_grad[1]: # input_hat
#         # grad_input_hat = mulb(grad_output, weight) # tmpmul.reshape(grad_output.shape)
        
# #         if context.needs_input_grad[3]: # running_mean : mu
#         # coef_m = (-1/(torch.sqrt(running_var + eps)))
#         # grad_running_mean = mulb(grad_input_hat, coef_m).sum(0).squeeze(0) #coef_m * grad_input_hat.sum(0).squeeze(0)
        

#         #if context.needs_input_grad[4]: # running_var : sigma
#         # coef = -0.5*(running_var + eps)**(-1.5)
#         # grad_running_var = mulb(grad_input_hat*(input-running_mean[None, :, None, None]), coef ).sum(0).squeeze(0)

#         #if context.needs_input_grad[0]: # input
#         coef_inp = Hadamard((1/N)*weight, (running_var + eps)**(-0.5))
#         # print('input_hat.shape', input_hat.shape, )
#         part1 = -Hadamard(input_hat, grad_weight)
#         # print('part1.shape' ,part1.shape, grad_bias.shape)

#         part2 = N*grad_output
#         # print('torch.ones_like(input).shape', torch.ones_like(input).shape, 'grad_bias[None,:].shape', grad_bias[None,:].shape)
#         part3 = -torch.einsum('nijk,oijk->nijk', torch.ones_like(input), grad_bias[None,:]).squeeze()
#         # print('part3.shape', part3.shape)
#         if len(coef_inp.shape) ==1:
#             coef_inp = coef_inp.unsqueeze(1).unsqueeze(2)
#         else:
#             coef_inp = coef_inp[None,:]
#         # print('coef_inp', coef_inp.shape, part1.shape, coef_inp.expand_as(part1).shape)
#         # print('part2.shape', part2.shape,'part3.shape', part3.shape,)
#         grad_input = coef_inp.expand_as(part1) * (part1 + part2 + part3)
#         # print('grad_input',grad_input.shape )             
#         # grad_weight.sum(0).permute(2,1,0).squeeze()
#         # grad_bias.permute(2,1,0).squeeze()

#         # if len(grad_weight.shape)>1:
#         #     grad_weight = grad_weight.permute(1,2,0)
#         # if len(grad_bias.shape)>1:
#         #     grad_bias = grad_bias.permute(1,2,0)
#         return grad_input, grad_weight.sum(dim=(-1, -2)), grad_bias.sum(dim=(-1, -2)), None, None, None, None