import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.nn import init
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _single, _pair
import copy

# In future, it's better to hijack the cpp implementation: https://github.com/pytorch/pytorch/blob/4f1f084d221a7ab9edbf09ab1904edde6d49848c/aten/src/THCUNN/generic/SpatialConvolutionMM.cu


class LinearAsymFunc(autograd.Function):
    @staticmethod
    # same as reference linear function, but with additional fa tensor for backward
    def forward(context, input, weight, weight_feedback, bias, algorithm_id, primitive_weights):

        context.save_for_backward(input, weight, weight_feedback, bias, Variable(torch.tensor(algorithm_id)), \
            Variable(torch.tensor(primitive_weights)))
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(context, grad_output):
        input, weight, weight_feedback, bias, algorithm_id, primitive_weights = context.saved_variables
        grad_input = grad_weight = grad_weight_feedback = grad_bias = None

        if context.needs_input_grad[0]:
            # all of the logic of FA resides in this one line
            # calculate the gradient of input with fixed fa tensor, rather than the "correct" model weight
            grad_input = grad_output.mm(weight_feedback)
        if context.needs_input_grad[1]:
            # grad for weight with FA'ed grad_output from downstream layer
            # it is same with original linear function
            grad_weight = grad_output.t().mm(input)
        if context.needs_input_grad[2]:
            if algorithm_id == 0:
                grad_weight_feedback = weight_feedback-weight_feedback
            elif algorithm_id == 1:

                # amp + null + decay (we use decay in optimizer for weight_feedback)
                xl = input
                xlp1T = input.mm(weight.t()).t()
                xlxlp1T = -xlp1T.mm(xl) 
                #print(torch.chain_matmul(weight_feedback,xlxlp1T.t(),xlxlp1T ).shape)
                alpha = primitive_weights[0] #0.3
                beta = primitive_weights[1] #0.02
                gamma = primitive_weights[2] #3*10e-6
                grad_weight_feedback = -alpha*xlxlp1T -gamma*torch.chain_matmul(weight_feedback,xlxlp1T.t(),xlxlp1T ) -beta*weight_feedback
         
        if bias is not None and context.needs_input_grad[3]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_weight_feedback, grad_bias, None, None



class LinearModule(nn.Module):
    """
    Implementation of a linear module which uses random feedback weights
    in its backward pass, as described in Lillicrap et al., 2016:

    https://www.nature.com/articles/ncomms13276
    """

    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=False, algorithm='FA', bottomup=1, primitive_weights=[0,0,0]):
        super(LinearModule, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features), requires_grad=True)
        self.weight_feedback = Parameter(torch.FloatTensor( out_features, in_features), requires_grad=True)
        self.bottomup = bottomup
        self.algorithm = algorithm
        self.primitive_weights =  primitive_weights
        if self.algorithm == 'FA':
            self.algorithm_id = 0
        elif self.algorithm.startswith('SL'):
            self.algorithm_id = 0
        elif self.algorithm ==  'IA':
            self.algorithm_id = 1
        
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
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

    def forward(self, inputs):
        
        if self.algorithm == 'BP':
            self.weight_feedback.data = copy.deepcopy(self.weight.data)
            
        if self.bottomup:
                
            if self.algorithm == 'BP':
                return F.linear(inputs, self.weight, self.bias)
            else:
                return LinearAsymFunc.apply(inputs, self.weight, self.weight_feedback, self.bias, self.algorithm_id, self.primitive_weights)
    
        else:
            
            if self.algorithm == 'BP':
                return F.linear(inputs, self.weight_feedback.t(), self.bias)
            else:
                return LinearAsymFunc.apply(inputs, self.weight_feedback.t(), self.weight.t(), self.bias, self.algorithm_id, self.primitive_weights)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )



class ConvAsymFunc(autograd.Function):
    @staticmethod
    # same as reference linear function, but with additional fa tensor for backward
    def forward(context, inputs, weight, weight_feedback, bias, stride, padding, groups, dilation, algorithm_id, primitive_weights):
        #note that only symmetric stride, padding, etc are supported for now
        context.save_for_backward(inputs, weight, weight_feedback, bias, Variable(torch.tensor(stride[0])),\
            Variable(torch.tensor(padding[0])), Variable(torch.tensor(groups)), Variable(torch.tensor(dilation[0])),\
            Variable(torch.tensor(algorithm_id)), Variable(torch.tensor(primitive_weights)))
        
        output = F.conv2d(inputs, weight, bias, stride, padding, dilation, groups)                
        return output

    @staticmethod
    def backward(context, grad_output):
        inputs, weight, weight_feedback, bias, stride, padding, groups, dilation, algorithm_id, primitive_weights = context.saved_tensors
        stride, padding, groups, dilation = stride.item(), padding.item(), groups.item(), dilation.item()
        grad_input = grad_weight = grad_weight_feedback = grad_bias = None
        grad_stride = grad_padding = grad_groups = grad_dilation = None

        if context.needs_input_grad[0]:
            
            grad_input = nn.grad.conv2d_input(inputs.shape, weight_feedback, grad_output, stride, padding, dilation, groups)
        
        if context.needs_input_grad[1]:
            # copied from nn.grad on master becuase of a wierd error of torch.narrow            
            stride = _pair(stride)
            padding = _pair(padding)
            dilation = _pair(dilation)
            in_channels = inputs.shape[1]
            out_channels = grad_output.shape[1]
            min_batch = inputs.shape[0]
            weight_size = weight.shape
            # print('before ravel:',inputs.shape, grad_output.shape, in_channels * min_batch)

            grad_output = grad_output.contiguous().repeat(1, in_channels // groups, 1, 1)
            grad_output = grad_output.contiguous().view(
                grad_output.shape[0] * grad_output.shape[1], 1, grad_output.shape[2],
                grad_output.shape[3])

            inputs_for_grad1 = inputs.contiguous().view(1, inputs.shape[0] * inputs.shape[1],
                                            inputs.shape[2], inputs.shape[3])
            # print('after ravel:',inputs.shape, grad_output.shape, in_channels * min_batch)

            grad_weight = torch.conv2d(inputs_for_grad1, grad_output, None, dilation, padding,
                                    stride, in_channels * min_batch)

            grad_weight = grad_weight.contiguous().view(
                min_batch, grad_weight.shape[1] // min_batch, grad_weight.shape[2],
                grad_weight.shape[3])

            grad_weight =  grad_weight.sum(dim=0).view(in_channels // groups, out_channels,grad_weight.shape[2], grad_weight.shape[3]).transpose(0, 1).narrow(2, 0, weight_size[2]).narrow(3, 0, weight_size[3])
            
        if context.needs_input_grad[2]:
            if algorithm_id == 0 : #FA
                grad_weight_feedback = weight_feedback-weight_feedback
            elif algorithm_id == 1: #Info_align   

                alpha = primitive_weights[0].to(weight.device) #0.3
                beta = primitive_weights[1].to(weight.device) #0.02
                gamma = primitive_weights[2].to(weight.device) #3*10e-6  
                # print('inputs:',inputs.type(), 'weight:',weight.type(), 'weight_feedback:',weight_feedback.type())
                             
                m1 = nn.Conv2d(weight.shape[1], weight.shape[0], weight.shape[2], bias=bias, stride=stride, padding=padding, groups=groups, dilation=dilation)
                bn1 = nn.BatchNorm2d(weight.shape[0], affine=False, track_running_stats=False)
                # insn1 = nn.InstanceNorm2d(weight.shape[1])
                m2 = nn.ConvTranspose2d(weight_feedback.shape[0], weight_feedback.shape[1], weight_feedback.shape[2], bias=bias, stride=stride, padding=padding, output_padding=0, groups=groups, dilation=dilation)
                # insn2 = nn.InstanceNorm2d(weight_feedback.shape[1])
                #F.normalize(dim=1)


                net_local = nn.Sequential(m1, bn1, nn.ReLU(), m2).to(weight.device)
                criterion_recons = nn.MSELoss()
                xl = inputs.detach().clone()
                xl.requires_grad = False
                xl = xl/torch.norm(xl)
                # print('xl',xl.shape)
                # [print(k, t.shape) for k,t in net_local.state_dict().items()]
                # print('-----------------') 
                # print('permuted 0.weight:', weight.permute(1, 0, 2, 3).shape,'3.weight:', weight_feedback.shape)

                state_dict = {'0.weight': weight, '3.weight': weight_feedback } #torch.flip
                
                net_local.load_state_dict(state_dict)

                # print('net_local', net_local)
                with torch.enable_grad():
                    output_local = net_local(xl)

                    output_local = output_local/ torch.norm(output_local)
                    loss_amp = criterion_recons(F.interpolate(output_local,size=xl.shape[-1]), xl)
                    loss_null = criterion_recons(output_local, torch.zeros_like(output_local).to(output_local.device))
                    loss_local = alpha * loss_amp + gamma * loss_null
                    
                    grad_weight_feedback_ampnull = autograd.grad(loss_local, net_local[3].weight, allow_unused=True)[0]
                    
                    
                
                grad_weight_feedback =  grad_weight_feedback_ampnull + beta * weight_feedback 
                # print(grad_weight_feedback[0,1, 0])
                # prnt('-----------------')
        if bias is not None and context.needs_input_grad[3]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_weight_feedback, grad_bias,grad_stride, grad_padding, grad_groups, grad_dilation, None, None


# class ConvTAsymFunc(autograd.Function):
#     @staticmethod
#     # same as reference linear function, but with additional fa tensor for backward
#     def forward(context, inputs, weight, weight_feedback, bias, stride, padding, output_padding, groups, dilation, algorithm_id):
#         #note that only symmetric stride, padding, etc are supported for now
        
#         context.save_for_backward(inputs, weight, weight_feedback, bias, Variable(torch.tensor(stride[0])), Variable(torch.tensor(padding[0])),Variable(torch.tensor(output_padding[0])), Variable(torch.tensor(groups)), Variable(torch.tensor(dilation[0])), Variable(torch.tensor(algorithm_id)))
        
#         output = F.conv_transpose2d(inputs, weight, bias, stride, padding, output_padding, groups, dilation)
                
#         return output

#     @staticmethod
#     def backward(context, grad_output):
#         inputs, weight, weight_feedback, bias, stride, padding, output_padding, groups, dilation, algorithm_id = context.saved_tensors
#         stride, padding, groups, dilation = stride.item(), padding.item(), groups.item(), dilation.item()
#         grad_input = grad_weight = grad_weight_feedback = grad_bias = None
#         grad_stride= grad_padding= grad_output_padding= grad_groups= grad_dilation = None

#         if context.needs_input_grad[0]:
#             # all of the logic of FA resides in this one line
#             # calculate the gradient of input with fixed fa tensor, rather than the "correct" model weight

#             grad_input = torch.conv2d(grad_output, weight, None, stride=stride, padding=padding, dilation=dilation,  groups=groups)
        
#         if context.needs_input_grad[1]:

#             stride = _pair(stride)
#             padding = _pair(padding)
#             dilation = _pair(dilation)
#             in_channels = inputs.shape[1]
#             out_channels = grad_output.shape[1]
#             min_batch = inputs.shape[0]
#             weight_size = weight.shape
#             print('before ravel:',inputs.shape, grad_output.shape, in_channels * min_batch)

#             grad_output = grad_output.contiguous().repeat(1, in_channels // groups, 1, 1)
#             grad_output = grad_output.contiguous().view(
#                 grad_output.shape[0] * grad_output.shape[1], 1, grad_output.shape[2],
#                 grad_output.shape[3])

#             inputs = inputs.contiguous().view(1, inputs.shape[0] * inputs.shape[1],
#                                             inputs.shape[2], inputs.shape[3])
#             print('after ravel:',inputs.shape, grad_output.shape, in_channels * min_batch)

#             grad_weight = F.conv2d(grad_output, inputs.permute(1,0,2,3), bias=None, stride=dilation, padding=padding,
#                                     dilation=stride, groups=in_channels * min_batch)

#             grad_weight = grad_weight.contiguous().view(
#                 min_batch, grad_weight.shape[1] // min_batch, grad_weight.shape[2],
#                 grad_weight.shape[3])

#             grad_weight =  grad_weight.sum(dim=0).view(in_channels // groups, out_channels,grad_weight.shape[2], grad_weight.shape[3]).transpose(0, 1).narrow(2, 0, weight_size[2]).narrow(3, 0, weight_size[3])
            
            
#             # ## modified from nn.grad
#             # ## change this for convT: grad_weight = F.grad.conv2d_weight(inputs, weight.shape, grad_output)
#             # stride = _pair(stride)
#             # padding = _pair(padding)
#             # dilation = _pair(dilation)
#             # in_channels = inputs.shape[1]
#             # out_channels = grad_output.shape[1]
#             # min_batch = inputs.shape[0]
#             # weight_size = weight.shape
#             # print('groups',groups)
#             # print('before ravel:',inputs.shape, grad_output.shape, in_channels * min_batch)

#             # grad_output = grad_output.contiguous().repeat(1, in_channels // groups, 1,1)
#             # grad_output_raveled = grad_output.contiguous().view(
#             #     grad_output.shape[0] * grad_output.shape[1], 1, grad_output.shape[2],
#             #     grad_output.shape[3])

#             # inputs_raveled = inputs.contiguous().view(1, inputs.shape[0] * inputs.shape[1],
#             #                                 inputs.shape[2], inputs.shape[3])

#             # ## grad_weight = torch.conv2d(inputs, grad_output, None, dilation, padding,
#             # #                         stride, in_channels * min_batch)

#             # #form functional: conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
#             # # torch.nn.functional.conv_transpose2d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1) → Tensor
#             # print(inputs_raveled.shape, grad_output_raveled.shape, in_channels * min_batch)
#             # grad_weight = F.conv_transpose2d(inputs_raveled, grad_output_raveled, bias=None, stride=stride, padding=padding, output_padding=0,
#             #                          groups=in_channels * min_batch, dilation=dilation)
            
#             # grad_weight = grad_weight.contiguous().view(
#             #     min_batch, grad_weight.shape[1] // min_batch, grad_weight.shape[2],
#             #     grad_weight.shape[3])

#             # grad_weight = grad_weight.sum(dim=0).view(
#             #     in_channels // groups, out_channels,
#             #     grad_weight.shape[2], grad_weight.shape[3]).transpose(0, 1).narrow(
#             #         2, 0, weight_size[2]).narrow(3, 0, weight_size[3])

            

#         if context.needs_input_grad[2]:
#             if algorithm_id == 0 : #FA
#                 grad_weight = weight_feedback-weight_feedback
#             elif algorithm_id == 1: #Info_align

#                 # first we compute the latent and detach it to have the input to the network
#                 # then we create a network to do the recostruction 
#                 latents = F.conv2d(inputs, weight, bias, stride, padding, dilation, groups)
#                 # weight and weight_feedback look swapped here but beccause we already swapped them in the module, we need to respect the order 
#                 m1 = nn.ConvTranspose2d(weight.shape[0], weight.shape[1], weight.shape[2], bias=bias, stride=stride, padding=padding, output_padding=0, groups=groups, dilation=dilation)
#                 m2 = nn.Conv2d(weight_feedback.shape[0], weight_feedback.shape[1], weight_feedback.shape[2], bias=bias, stride=stride, padding=padding, groups=groups, dilation=dilation)
#                 net_local = nn.Sequential(m1, m2)
#                 criterion_recons = nn.MSELoss()
#                 xl = latents.detach().clone()
#                 xl.requires_grad = False

#                 state_dict = {'0.weight': weight_feedback, '1.weight': weight }
#                 net_local.load_state_dict(state_dict)

#                 with torch.enable_grad():
#                     output_local = net_local(xl)
#                     loss_local = criterion_recons(output_local, inputs)
#                     grad_weight_recons = autograd.grad(loss_local, net_local[1].weight, allow_unused=True)[0]
        
#                 alpha = 1
#                 grad_weight = alpha*grad_weight_recons
        
#         if bias is not None and context.needs_input_grad[3]:
#             grad_bias = grad_output.sum(0).squeeze(0)

#         return grad_input, grad_weight, grad_weight_feedback, grad_bias,grad_stride, grad_padding, grad_output_padding,  grad_groups, grad_dilation, None

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
                 groups, bias, padding_mode, algorithm='FA', bottomup=1):
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
        self.bottomup = bottomup

        if transposed:
            self.weight = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size), requires_grad=True)

            self.weight_feedback = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size), requires_grad=True)
        else:
            self.weight = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size), requires_grad=True)

            self.weight_feedback = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size), requires_grad=True)
        
        
        if self.algorithm == 'BP':
            self.weight_feedback = copy.deepcopy(self.weight)
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_feedback, a=math.sqrt(5))
        if self.algorithm == 'BP':
            self.weight_feedback = copy.deepcopy(self.weight)
#             pass
#         else:
#             init.kaiming_uniform_(self.weight_feedback, a=math.sqrt(5))

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
                 bias=False, padding_mode='zeros', algorithm='FA', primitive_weights=[0,0,0], bottomup=1):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        if bottomup == 0:
            self.transposed = True
        else:
            self.transposed = False

        super(AsymmetricFeedbackConv2d, self).__init__(
            in_channels , out_channels, kernel_size, stride, padding, dilation,
            self.transposed, _pair(0), groups, bias, padding_mode, algorithm, bottomup)
        
        self.primitive_weights = primitive_weights
        self.algorithm = algorithm
        if self.algorithm == 'FA':
            self.algorithm_id = 0
        elif self.algorithm.startswith('SL'):
            self.algorithm_id = 1
        elif self.algorithm ==  'IA':
            self.algorithm_id = 1
        

    def forward(self, input):
#         if self.padding_mode == 'circular':
#             expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
#                                 (self.padding[0] + 1) // 2, self.padding[0] // 2)
#             return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
#                             self.weight, self.bias, self.stride,
#                             _pair(0), self.dilation, self.groups)
#        if self.transposed:
# #             # The elegant way is to find the proper way to calculate output padding, setting stride=1 is just a hack
# #             if self.stride[0] >1:               
# #                 output_padding = 0
# #             else:
# #                 output_padding = 0
# #             # stride = 1#(self.stride[0]+1,self.stride[1]+1)
#             output_padding = 0
#             return F.conv_transpose2d(
#                 input, self.weight_feedback.permute(1,0,2,3), self.bias, self.stride, self.padding,
#                 output_padding, self.groups, self.dilation)
        
# #         F.conv_transpose2d(
# #                 input, self.weight.permute(1,0,2,3), self.bias, self.stride, self.padding,
# #                 output_padding, self.groups, self.dilation)


        if self.transposed:
            
            if self.algorithm == 'BP':
                output_padding = 0
                
                return F.conv_transpose2d(input, self.weight.permute(1,0,2,3), self.bias, self.stride, self.padding,
                output_padding, self.groups, self.dilation)
            else:
                return ConvTAsymFunc.apply(input, self.weight_feedback.permute(1,0,2,3), self.weight_feedback, \
                self.bias, self.stride, self.padding, self.groups, self.dilation, self.algorithm_id)

        else:
            
            if self.algorithm == 'BP':
                self.weight_feedback.data = copy.deepcopy(self.weight.data)
                return F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
            else:
                return ConvAsymFunc.apply(input, self.weight, self.weight_feedback ,self.bias, self.stride,\
                self.padding, self.groups, self.dilation, self.algorithm_id, self.primitive_weights)



# class AsymmetricFeedbackConvTranspose2d(_ConvNdFA):
#     """
#     Implementation of a 2D convolution module which uses random feedback weights
#     in its backward pass, as described in Lillicrap et al., 2016:

#     https://www.nature.com/articles/ncomms13276
#     """

#     def __init__(self, in_channels, out_channels, kernel_size, stride=1,
#                  padding=0, output_padding=0, dilation=1, groups=1,
#                  bias=False, padding_mode='zeros', algorithm='FA', bottomup=0):
#         kernel_size = _pair(kernel_size)
#         stride = _pair(stride)
#         padding = _pair(padding)
#         output_padding = _pair(output_padding)
#         dilation = _pair(dilation)
#         if bottomup == 0:
#             self.transposed = True
#         else:
#             self.transposed = False

#         super(AsymmetricFeedbackConvTranspose2d, self).__init__(
#             in_channels , out_channels, kernel_size, stride, padding, dilation,
#             self.transposed, output_padding, groups, bias, padding_mode, algorithm, bottomup)
        
        
#         self.algorithm = algorithm
#         if self.algorithm in 'FA':
#             self.algorithm_id = 0
#         elif self.algorithm.startswith('SL'):
#             self.algorithm_id = 0
#         elif self.algorithm ==  'IA':
#             self.algorithm_id = 1
        

#     def forward(self, input):
# #         if self.padding_mode == 'circular':
# #             expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
# #                                 (self.padding[0] + 1) // 2, self.padding[0] // 2)
# #             return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
# #                             self.weight, self.bias, self.stride,
# #                             _pair(0), self.dilation, self.groups)
# #        if self.transposed:
# # #             # The elegant way is to find the proper way to calculate output padding, setting stride=1 is just a hack
# # #             if self.stride[0] >1:               
# # #                 output_padding = 0
# # #             else:
# # #                 output_padding = 0
# # #             # stride = 1#(self.stride[0]+1,self.stride[1]+1)
# #             output_padding = 0
# #             return F.conv_transpose2d(
# #                 input, self.weight_feedback.permute(1,0,2,3), self.bias, self.stride, self.padding,
# #                 output_padding, self.groups, self.dilation)
        
# # #         F.conv_transpose2d(
# # #                 input, self.weight.permute(1,0,2,3), self.bias, self.stride, self.padding,
# # #                 output_padding, self.groups, self.dilation)
        
#         # if self.transposed:
#         if self.algorithm == 'BP':
#             # output_padding = 0
            
#             return F.conv_transpose2d(input, self.weight_feedback, self.bias, self.stride, self.padding,
#             self.output_padding, self.groups, self.dilation)
#         else:
#             return ConvTAsymFunc.apply(input, self.weight_feedback, self.weight ,self.bias, self.stride, self.padding, self.output_padding, self.groups, self.dilation, self.algorithm_id)

#         # else:
#         #     if self.algorithm == 'BP':
#         #         return F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
#         #     else:
#         #         return ConvAsymFunc.apply(input, self.weight, self.weight_feedback ,self.bias, self.stride, self.padding, self.groups, self.dilation, self.algorithm_id)


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


class ReLUBFunc(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input, criteria):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.

        Here, criteria is the input to the module ReLU in forward pass
        """
        ctx.save_for_backward(criteria)
        output = input.clone()
        output[criteria<0] = 0
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        criteria, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[criteria < 0] = 0
        return grad_input, None

ReLUB = ReLUBFunc.apply
