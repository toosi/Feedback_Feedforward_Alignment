"""
This module contains linear layer modules with nonsymmetric backward weights,
so I defined a customized autograd fucntion for a linear layer.
Since I wanted to copy the weights between the forward and backward paths,
I created two linear modules which their only difference is the name of the forward
and backward weights:
LinearUp     forward: weight_up        backward:weight_down
LinearDown   forward: weight_down      backward:weight_up

Currently, Bias is set to False, because the exchanging state_dics was challenging
This code is heavily inpired by L0SG's implementation
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import autograd
from torch.autograd import Variable
import math
from torch._thnn import type2backend
import copy
# print('******* in customized_modules',torch.__version__,'*******')

# class LinearSymmFunction(autograd.Function):

#     """
#     Autograd function for a linear layer with asymmetric feedback and feedforward pathways
#     bias is set to None for now
#     """

#     @staticmethod
#     # same as reference linear function, but with additional fa tensor for backward
#     def forward(context, input, weight, bias=None):
#         context.save_for_backward(input, weight, bias)
#         output = input.mm(weight.t())
#         if bias is not None:
#             output += bias.unsqueeze(0).expand_as(output)
#         return output

#     @staticmethod
#     def backward(context, grad_output):
#         input, weight, bias = context.saved_variables
#         grad_input = grad_weight = grad_bias = None

#         if context.needs_input_grad[0]:
#             # all of the logic of FA resides in this one line
#             # calculate the gradient of input with fixed fa tensor, rather than the "correct" model weight
#             grad_input = grad_output.mm(weight)
#         if context.needs_input_grad[1]:
#             # grad for weight with FA'ed grad_output from downstream layer
#             # it is same with original linear function
#             grad_weight = grad_output.t().mm(input)
#         if bias is not None and context.needs_input_grad[3]:
#             grad_bias = grad_output.sum(0).squeeze(0)

#         return grad_input, grad_weight, grad_bias

#         return grad_input, grad_weight, grad_bias

# class LinearSymmModule(nn.Module):

#     """
#     a linear layer with symmetric feedback and feedforward pathways
#     """

#     def __init__(self, input_features, output_features, bias=False):     # we ignore bias for now
#         super(LinearSymmModule, self).__init__()
#         self.input_features = input_features
#         self.output_features = output_features

#         # weight and bias for forward pass
#         # weight has transposed form; more efficient (so i heard) (transposed at forward pass)
#         self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
#         if bias:
#             self.bias = nn.Parameter(torch.Tensor(output_features))
#         else:
#             self.register_parameter('bias', None)

#         # weight initialization
#         torch.nn.init.kaiming_uniform_(self.weight)
#         if bias:
#             torch.nn.init.constant(self.bias, 1)

#     def forward(self, input):
#         return LinearSymmFunction.apply(input, self.weight, self.bias)



#-----------------------------------------------------------

# class LinearYYFunction(autograd.Function):

#     """
#     Autograd function for a linear layer with asymmetric feedback and feedforward pathways
#     forward  : weight
#     backward : weight_feedback

#     bias is set to None for now
#     """

#     @staticmethod
#     # same as reference linear function, but with additional fa tensor for backward
#     def forward(context, input, weight, weight_feedback, bias=None):
#         context.save_for_backward(input, weight, weight_feedback, bias)
#         output = input.mm(weight.t())
#         if bias is not None:
#             output += bias.unsqueeze(0).expand_as(output)
#         return output

#     @staticmethod
#     def backward(context, grad_output):
#         input, weight, weight_feedback, bias = context.saved_variables
#         grad_input = grad_weight = grad_weight_feedback = grad_bias = None

#         if context.needs_input_grad[0]:
#             # all of the logic of FA resides in this one line
#             # calculate the gradient of input with fixed fa tensor, rather than the "correct" model weight

#             # to include nonlinearity in the backward path of YY uncomment this
#             # grad_input = nn.functional.tanh(grad_output.mm(weight_feedback))
#             grad_input = grad_output.mm(weight_feedback)
#         if context.needs_input_grad[1]:
#             # grad for weight with FA'ed grad_output from downstream layer
#             # it is same with original linear function
#             grad_weight = grad_output.t().mm(input)
#         if bias is not None and context.needs_input_grad[3]:
#             grad_bias = grad_output.sum(0).squeeze(0)

#         return grad_input, grad_weight, grad_weight_feedback, grad_bias



class LinearFunction(autograd.Function):

    """
    Autograd function for a linear layer with asymmetric feedback and feedforward pathways
    forward  : weight
    backward : weight_feedback

    bias is set to None for now
    """

    @staticmethod
    # same as reference linear function, but with additional fa tensor for backward
    def forward(context, input, weight, weight_feedback, bias=None):
        context.save_for_backward(input, weight, weight_feedback, bias)
        output = input.mm(weight.t())
        if bias is not None:
            output  += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(context, grad_output):
        input, weight, weight_feedback, bias = context.saved_variables
        grad_input = grad_weight = grad_weight_feedback = grad_bias = None

        if context.needs_input_grad[0]:
            # all of the logic of FA resides in this one line
            # calculate the gradient of input with fixed fa tensor, rather than the "correct" model weight
            grad_input = grad_output.mm(weight_feedback)
        if context.needs_input_grad[1]:
            # grad for weight with FA'ed grad_output from downstream layer
            # it is same with original linear function
            grad_weight = grad_output.t().mm(input)
        if bias is not None and context.needs_input_grad[3]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_weight_feedback, grad_bias

class LinearModule(nn.Linear):

    """
    a linear layer with asymmetric feedback and feedforward pathways
    forward  : weight
    backward : weight_feedback
    """

    def __init__(self, *args, algorithm, **kwargs):     # we ignore bias for now

        assert algorithm in ('YY','FA', 'BP'), 'feedback algorithm %s is not implemented'

        super(LinearModule, self).__init__(*args,*kwargs)
        # self.input_features = input_features
        # self.output_features = output_features
        self.algorithm = algorithm

        # weight and bias for forward pass
        # weight has transposed form for efficiency (?) (transposed at forward pass)

        # self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        # as in torchvision/nn/modules/linear scaling was based on weight input (weight.size(1))
        # since  weight_feedback is the transpose scaling should be like below
        self.scale_feedback = 1. / math.sqrt(self.weight.size(0))
        weight_feedback = None
        # if bias:
        #     self.bias = nn.Parameter(torch.Tensor(output_features))
        # else:
        self.register_parameter('bias', None)

        #weight_feedback = torch.Tensor(output_features, input_features)
        weight_feedback = self.weight.new_empty(self.weight.shape).detach()
        torch.nn.init.uniform_(weight_feedback.cuda(), -self.scale_feedback, self.scale_feedback)
        # self.reset_parameters()
        if algorithm != 'BP':
            self.register_buffer('weight_feedback', weight_feedback)

        
    # def reset_parameters(self):

    #     # weight initialization
    #     torch.nn.init.kaiming_uniform_(self.weight)
    #     torch.nn.init.kaiming_uniform_(self.weight_feedback)
    #     if self.bias is not None:
    #         torch.nn.init.constant(self.bias, 1)



    def forward(self, input):

        # if self.algorithm == 'FA':

        #     weight_feedback = self.weight_feedback
        if self.algorithm == 'BP':

            self.weight_feedback = copy.deepcopy(self.weight.detach())

            return LinearFunction.apply(input, self.weight, self.weight_feedback, self.bias)
        elif self.algorithm == 'FA':

            self.weight_feedback = self.weight_feedback.detach().clone()
            return LinearFunction.apply(input, self.weight, self.weight_feedback, self.bias)
        # elif self.algorithm == 'YY':

        #     return LinearFunction.apply(input, self.weight, self.weight_feedback, self.bias)




#-----------------------------------------------------------------------------------------------

"""
Testing conv2d, modified from https://github.com/willwx/sign-symmetry/

useful stuff for implementation of transposeconv:
https://github.com/torch/nn/blob/master/SpatialFullConvolution.lua
https://github.com/tylergenter/pytorch/blob/master/torch/legacy/nn/SpatialDilatedConvolution.py
https://github.com/jcjohnson/pytorch-old/blob/master/torch/legacy/nn/SpatialFullConvolution.py
https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/Convolution.cpp#L289-L297
https://github.com/pytorch/pytorch/tree/master/torch/_thnn
https://discuss.pytorch.org/t/forward-and-backward-implementation-of-max-pool-2d/12681
halochou/conv2dlocal.py

"""

import torch.autograd as autograd
from torch._thnn import type2backend
# print(torch._thnn.__file__)

class AsymmetricFeedbackConv2dFunc(autograd.Function):

    @staticmethod
    def forward(context, input, weight, weight_feedback, bias, stride, padding):
        _backend = type2backend[input.type()]
        input = input.contiguous()
        output = input.new()
        finput = input.new()
        fgradInput = input.new()

        _backend.SpatialConvolutionMM_updateOutput(
            _backend.library_state,
            input,
            output,
            weight,
            bias,
            finput,
            fgradInput,
            weight.shape[2], weight.shape[3],
            int(stride[0]), int(stride[1]),
            int(padding[0]), int(padding[1])
        )

        context.save_for_backward(input, weight, weight_feedback, bias, stride, padding, finput, fgradInput)
        return output

    @staticmethod
    def backward(context, grad_output):
        input, weight, weight_feedback, bias, stride, padding, finput, fgradInput = context.saved_variables
        grad_input = grad_weight = grad_bias = None

        _backend = type2backend[grad_output.type()]
        grad_output = grad_output.contiguous()
        ksize = tuple(weight.shape[-2:])

        if context.needs_input_grad[0]:
            grad_input = input.new()

            _backend.SpatialConvolutionMM_updateGradInput(
                _backend.library_state,
                input,
                grad_output,
                grad_input,
                weight_feedback,
                finput,
                fgradInput,
                ksize[0], ksize[1],
                int(stride[0]), int(stride[1]),
                int(padding[0]), int(padding[1])
            )

        if context.needs_input_grad[1] or context.needs_input_grad[3]:
            grad_weight = weight.grad or weight.new_zeros(weight.shape)
            grad_bias = None if bias is None else bias.grad or bias.new_zeros(bias.shape)

            _backend.SpatialConvolutionMM_accGradParameters(
                _backend.library_state,
                input,
                grad_output,
                grad_weight,
                grad_bias,
                finput,
                fgradInput,
                ksize[0], ksize[1],
                int(stride[0]), int(stride[1]),
                int(padding[0]), int(padding[1]),
                1
            )

        return grad_input, grad_weight, None, grad_bias, None, None




class AsymmetricFeedbackConv2d(nn.Conv2d):
    def __init__(self, *args,algorithm='FA', **kwargs):

        assert algorithm in ('YY','FA', 'BP'), 'feedback algorithm %s is not implemented'

        super(AsymmetricFeedbackConv2d, self).__init__(*args, **kwargs)

        # this scale is used to initialize resnet models in torchvision/models/resnet.py
        # here it is used as a heuristic to avoid the exploding gradient problem
        self.scale = math.sqrt(2 / (self.kernel_size[0] * self.kernel_size[1] * self.out_channels))
        self.algorithm = algorithm
        self.register_parameter('bias', None)


        # weight_feedback = None
        # weight_feedback = self.weight.new_empty(self.weight.shape).detach_()
        # weight_feedback.data.normal_(0, self.scale)
        if 'stride' in kwargs.keys():
            stride = kwargs['stride']
        else:
            stride = 1

        if 'padding' in kwargs.keys():
            padding= kwargs['padding']
        else:
            padding = 0
            
        self.stride_tensor = torch.Tensor((stride, stride)).type(torch.int)
        self.padding_tensor = torch.Tensor((padding, padding)).type(torch.int)
        
        # if algorithm != 'BP':
        #     self.register_buffer('weight_feedback', weight_feedback)
        #     # save tensor version of stride & padding for use in ws_conv2d_function

        #     self.register_buffer('stride_tensor', self.stride_tensor)
        #     self.register_buffer('padding_tensor', self.padding_tensor)
        #     self.reset_weight_feedback()

        # save tensor version of stride & padding for use in ws_conv2d_function


        # self.reset_weight_feedback()
        weight_feedback = None
        weight_feedback = self.weight.new_empty(self.weight.shape).detach_()
        weight_feedback.data.normal_(0, self.scale)

        self.register_buffer('weight_feedback', weight_feedback)
    

    # def reset_weight_feedback(self):

    #     weight_feedback = None
    #     if self.algorithm in ['FA','YY']:

    #         weight_feedback = self.weight.new_empty(self.weight.shape).detach_()
    #         weight_feedback.data.normal_(0, self.scale)

    #     self.register_buffer('weight_feedback', weight_feedback)



    def forward(self, input):

        if self.algorithm == 'FA':
            self.weight_feedback = self.weight_feedback
        elif self.algorithm == 'YY':
            self.weight_feedback = self.weight_feedback
        elif self.algorithm == 'BP':
            self.weight_feedback = self.weight.detach()
        else:
            raise RuntimeError('Unsupported algorithm for %s'%self.__class__.__name__)

        return AsymmetricFeedbackConv2dFunc.apply(
            input, self.weight, self.weight_feedback, self.bias, self.stride_tensor, self.padding_tensor)

#-----------------------------------------------------------------------------


class AsymmetricFeedbackConvTranspose2dFunc(autograd.Function):

    @staticmethod
    def forward(context, input, weight, weight_feedback, bias, stride, padding, output_padding):
        _backend = type2backend[input.type()]
        input = input.contiguous()
        output = input.new()
        finput = input.new()
        fgradInput = input.new()
        kW , kH = weight.shape[2], weight.shape[3]
        dW, dH = int(stride[0]), int(stride[1])
        padW, padH = int(padding[0]), int(padding[1]),
        adjW, adjH = int(output_padding[0]), int(output_padding[1])

        # The output width:
        # owidth  = (width  - 1) * dW - 2*padW + kW + adjW

        _backend.SpatialFullConvolution_updateOutput(
            _backend.library_state,
            input,
            output,
            weight,
            bias,
            finput,
            fgradInput,
            kW , kH,
            dW, dH,
            padW, padH,
            adjW, adjH
        )

        context.save_for_backward(input, weight, weight_feedback, bias, stride, padding, output_padding, finput, fgradInput)
        return output



    @staticmethod
    def backward(context, grad_output):
        input, weight, weight_feedback, bias, stride, padding, output_padding, finput, fgradInput = context.saved_variables
        grad_input = grad_weight = grad_bias = None

        _backend = type2backend[grad_output.type()]
        grad_output = grad_output.contiguous()
        kW , kH = weight.shape[2], weight.shape[3]
        dW, dH = int(stride[0]), int(stride[1])
        padW, padH = int(padding[0]), int(padding[1]),
        adjW, adjH = int(output_padding[0]), int(output_padding[1])
        scale = 1


        if context.needs_input_grad[0]:
            grad_input = input.new()

            _backend.SpatialFullConvolution_updateGradInput(
                _backend.library_state,
                input,
                grad_output,
                grad_input,
                weight_feedback,
                finput,
                kW , kH,
                dW, dH,
                padW, padH,
                adjW, adjH
            )

        if context.needs_input_grad[1] or context.needs_input_grad[3]:
            grad_weight = weight.grad or weight.new_zeros(weight.shape)
            grad_bias = None if bias is None else bias.grad or bias.new_zeros(bias.shape)

            _backend.SpatialFullConvolution_accGradParameters(
                _backend.library_state,
                input,
                grad_output,
                grad_weight,
                grad_bias,
                finput,
                fgradInput,
                kW , kH,
                dW, dH,
                padW, padH,
                adjW, adjH,
                scale
            )



        return grad_input, grad_weight, None, grad_bias, None, None, None




class AsymmetricFeedbackConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, *args, algorithm='FA', **kwargs):

        assert algorithm in ('YY','FA', 'BP'), 'feedback algorithm %s is not implemented'

        super(AsymmetricFeedbackConvTranspose2d, self).__init__(*args, **kwargs)

        # this scale is used to initialize resnet models in torchvision/models/resnet.py
        # here it is used as a heuristic to avoid the exploding gradient problem
        self.scale = math.sqrt(2 / (self.kernel_size[0] * self.kernel_size[1] * self.out_channels))
        self.algorithm = algorithm
        self.register_parameter('bias', None)

        
        weight_feedback = None
        weight_feedback = self.weight.new_empty(self.weight.shape).detach_()
        weight_feedback.data.normal_(0, self.scale)

        if 'stride' in kwargs.keys():
            stride = kwargs['stride']
        else:
            stride = 1

        if 'padding' in kwargs.keys():
            padding= kwargs['padding']
        else:
            padding = 0

        if 'output_padding' in kwargs.keys():
            output_padding= kwargs['output_padding']
        else:
            output_padding = 0

        self.stride_tensor = torch.Tensor((stride, stride)).type(torch.int)
        self.padding_tensor = torch.Tensor((padding, padding)).type(torch.int)
        self.output_padding_tensor = torch.Tensor((output_padding, output_padding)).type(torch.int)
        # # save tensor version of stride & padding for use in ws_conv2d_function
        # self.stride_tensor = torch.Tensor(self.stride).type(torch.int)
        # self.padding_tensor = torch.Tensor(self.padding).type(torch.int)
        # self.output_padding_tensor = torch.Tensor(self.output_padding).type(torch.int)
        
        # if algorithm != 'BP':
        #     self.register_buffer('weight_feedback', weight_feedback)
            
        #     self.register_buffer('stride_tensor', self.stride_tensor)
        #     self.register_buffer('padding_tensor', self.padding_tensor)
        #     self.register_buffer('output_padding_tensor', self.padding_tensor)
        #     self.reset_weight_feedback()

        if algorithm != 'BP':
            self.register_buffer('weight_feedback', weight_feedback)

    # def reset_weight_feedback(self):
    #     weight_feedback = None
    #     if self.algorithm in ['FA','YY']:
    #         weight_feedback = self.weight.new_empty(self.weight.shape).detach_()
    #         weight_feedback.data.normal_(0, self.scale)
    #     self.register_buffer('weight_feedback', weight_feedback)



    def forward(self, input):
        if self.algorithm == 'FA':
            self.weight_feedback = self.weight_feedback
        elif self.algorithm == 'YY':
            self.weight_feedback = self.weight_feedback
        elif self.algorithm == 'BP':
            self.weight_feedback = self.weight.detach()
        else:
            raise RuntimeError('Unsupported algorithm for %s'%self.__class__.__name__)

        return AsymmetricFeedbackConvTranspose2dFunc.apply(
            input, self.weight, self.weight_feedback, self.bias, self.stride_tensor, self.padding_tensor, self.output_padding_tensor)


#--------------------------------------------------------------


# # https://github.com/jcjohnson/pytorch-old/blob/master/torch/legacy/nn/BatchNormalization.py

# class AsymmetricFeedbackBatchNormFunc(autograd.Function):

#     @staticmethod
#     def forward(context, input, weight, weight_feedback, bias, stride, padding):
#         _backend = type2backend[input.type()]
#         input = input.contiguous()
#         output = input.new()
#         finput = input.new()
#         fgradInput = input.new()

#         _backend.BatchNormalization_updateOutput(
#             _backend.library_state,
#             input,
#             output,
#             weight,
#             bias,
#             running_mean,running_var,
#             save_mean, save_std,
#             train, 
#             momentum,
#             eps
#         )

#         context.save_for_backward(input, weight, weight_feedback, bias, stride, padding, finput, fgradInput)
#         return output

#     @staticmethod
#     def backward(context, grad_output):
#         input, weight, weight_feedback, bias, stride, padding, finput, fgradInput = context.saved_variables
#         grad_input = grad_weight = grad_bias = None

#         _backend = type2backend[grad_output.type()]
#         grad_output = grad_output.contiguous()
#         ksize = tuple(weight.shape[-2:])

#         if context.needs_input_grad[0]:
#             grad_input = input.new()

#             _backend.BatchNormalization_backward(
#                 _backend.library_state,
#                 input,
#                 grad_output,
#                 grad_input,
#                 weight_feedback,
#                 finput,
#                 fgradInput,
#                 ksize[0], ksize[1],
#                 int(stride[0]), int(stride[1]),
#                 int(padding[0]), int(padding[1])
#             )

#         if context.needs_input_grad[1] or context.needs_input_grad[3]:
#             grad_weight = weight.grad or weight.new_zeros(weight.shape)
#             grad_bias = None if bias is None else bias.grad or bias.new_zeros(bias.shape)

#             _backend.SpatialConvolutionMM_accGradParameters(
#                 _backend.library_state,
#                 input,
#                 grad_output,
#                 grad_weight,
#                 grad_bias,
#                 finput,
#                 fgradInput,
#                 ksize[0], ksize[1],
#                 int(stride[0]), int(stride[1]),
#                 int(padding[0]), int(padding[1]),
#                 1
#             )

#         return grad_input, grad_weight, None, grad_bias, None, None




# class AsymmetricFeedbackBatchNorm(nn.Conv2d):
#     def __init__(self, *args,algorithm='FA', **kwargs):

#         assert algorithm in ('YY','FA', 'BP'), 'feedback algorithm %s is not implemented'

#         super(AsymmetricFeedbackBatchNorm, self).__init__(*args, **kwargs)

#         # this scale is used to initialize resnet models in torchvision/models/resnet.py
#         # here it is used as a heuristic to avoid the exploding gradient problem
#         self.scale = math.sqrt(2 / (self.kernel_size[0] * self.kernel_size[1] * self.out_channels))
#         self.algorithm = algorithm


#         # save tensor version of stride & padding for use in ws_conv2d_function
#         stride_tensor = torch.Tensor(self.stride).type(torch.int)
#         padding_tensor = torch.Tensor(self.padding).type(torch.int)
#         self.register_buffer('stride_tensor', stride_tensor)
#         self.register_buffer('padding_tensor', padding_tensor)
#         # self.reset_weight_feedback()
#         weight_feedback = None
#         weight_feedback = self.weight.new_empty(self.weight.shape).detach_()
#         weight_feedback.data.normal_(0, self.scale)

#         self.register_buffer('weight_feedback', weight_feedback)


#     # def reset_weight_feedback(self):

#     #     weight_feedback = None
#     #     if self.algorithm in ['FA','YY']:

#     #         weight_feedback = self.weight.new_empty(self.weight.shape).detach_()
#     #         weight_feedback.data.normal_(0, self.scale)

#     #     self.register_buffer('weight_feedback', weight_feedback)



#     def forward(self, input):

#         if self.algorithm == 'FA':
#             self.weight_feedback = self.weight_feedback
#         elif self.algorithm == 'YY':
#             self.weight_feedback = self.weight_feedback
#         elif self.algorithm == 'BP':
#             self.weight_feedback = self.weight.detach()
#         else:
#             raise RuntimeError('Unsupported algorithm for %s'%self.__class__.__name__)

#         return AsymmetricFeedbackBatchNormFunc.apply(
#             input, self.weight, self.weight_feedback, self.bias, self.stride_tensor, self.padding_tensor)
