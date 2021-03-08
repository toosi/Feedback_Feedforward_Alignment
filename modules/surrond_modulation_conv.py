# written by: jon-tow in https://github.com/jon-tow/surround-modulation-cnn/blob/master/model.py
from typing import Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

π = math.pi


# Gaussian Kernels.


def gaussian1d(kernel_size: int, σ: float) -> torch.Tensor:
    """
    Returns a 1D Gaussian kernel of the specified kernel size with spread, σ.
    Arguments:
    - kernel_size: The size of the resulting 1D Gaussian kernel.
    - σ: The standard deviation of the Gaussian.
    """
    assert kernel_size % 2 != 0, f'Kernel size must be an odd integer > 0. Got: {kernel_size}'
    assert σ > 0, f'Standard deviation must be non-negative. Got: {σ}'

    center = kernel_size // 2
    x = torch.arange(0, kernel_size, dtype=torch.float32)
    normalization = 1 / (σ * torch.sqrt(torch.tensor(2 * π)))
    kernel = normalization * torch.exp(-(x - center)**2 / (2 * σ**2))
    return kernel / kernel.sum()


def gaussian2d(kernel_size: Tuple[int, int], σ: float) -> torch.Tensor:
    """
    Returns a 2D Gaussian kernel constructed by multiplying two 1D Gaussian
    kernels of the specified kernel size with spread, σ.
    Arguments:
    - kernel_sizes: A 2-tuple of kernel sizes for the height and width spatial
                    dimensions of the resulting 2D Gaussian:
                    `(height, width)`
    - σ: The standard deviation of the 2D Gaussian.
    """
    height, width = kernel_size[0], kernel_size[1]
    kernel_x = gaussian1d(width, σ)
    kernel_y = gaussian1d(height, σ)
    # Note: `torch.ger` is the outer product (weird name from BLAS...).
    kernel = torch.ger(kernel_x, kernel_y)  # By separability of 1D Gaussians.
    return kernel


def DoG(kernel_size: Tuple[int, int], σs: Tuple[float, float]) -> torch.Tensor:
    """
    Returns the Difference of Gaussians (DoG) between two Gaussian kernels formed
    by the given standard deviations.
    See equation `(1)` of the paper.
    Arguments:
    - kernel_size: A 2-tuple of the kernel sizes for the height and width spatial
                   dimensions of the resulting Differece of Gaussians:
                   `(height, width)`
    - σs: A 2-tuple of standard deviations used to construct the two 2D Gaussian
          kernels.
    """
    σ1, σ2 = σs
    kernel = gaussian2d(kernel_size, σ1) - gaussian2d(kernel_size, σ2)
    return kernel


# Surround Modulation Kernel and Convolution Module.


def surround_modulation(kernel_size: int, σ_e: float, σ_i: float) -> torch.Tensor:
    """
    Returns a Surround Modulation (`sm`) kernel tensor with the given
    excitatory and inhibitory standard deviations.
    See equation `(2)` of the paper.
    Arguments:
    - kernel_size: The size of the Surround Modulation kernel. The size must be
                   an odd integer > 0 so that the surround modulation kernel can
                   have shape (2k + 1) x (2k + 1), where `k` is the padding size
                   used on an activation input.
    - σ_e: The standard deviation of the excitatory Gaussian.
    - σ_i: The standard deviation of the inhibitory Gaussian.
    """
    assert kernel_size % 2 != 0, f'The kernel size must be odd. Got: {kernel_size}'

    center = kernel_size // 2
    dog = DoG((kernel_size, kernel_size), σs=(σ_e, σ_i))
    kernel = dog / dog[center][center]
    return kernel


class SMConv(nn.Module):
    """ A module for Surround Modulation (SM) Convolutions. """

    def __init__(
        self,
        kernel_size: int = 5,
        σ_e: float = 1.2,
        σ_i: float = 1.4,
        mask_p: float = 0.5
    ):
        """
        Creates a Surround Modulation Convolution layer.
        _Note_: This module only works with 2D convolutions.
        Arguments:
        - kernel_size: The size of the Surround Modulation kernel. The size must
                       be an odd integer > 0 so that the surround modulation
                       kernel can have shape (2k + 1) x (2k + 1), where `k` is
                       where `k` is the padding size used on an activation
                       input.
        - σ_e: The standard deviation of the excitatory Gaussian.
               Default: 1.2
        - σ_i: The standard deviation of the inhibitory Gaussian.
               Default: 1.4
        - mask_p: The approximate percent of activation maps to mask. Approximate
               because the selected indices are not sampled uniquely. A large
               percent means NOT modulating a large number of maps in the input.
               Default: 0.5
        """
        assert mask_p >= 0.0 and mask_p <= 1.0, \
            f'The percent of activation maps to mask must lie in [0.0, 1.0]. Got: {mask_p}'

        super().__init__()
        self.kernel_size = kernel_size
        self.σ_e = σ_e
        self.σ_i = σ_i
        self.mask_p = mask_p
        # Convolution padding size to keep spatial dimensions the same.
        self.same = self.kernel_size // 2
        self.register_buffer(
            'kernel', surround_modulation(kernel_size, σ_e, σ_i))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Returns a tensor of activation maps with `mask_p` percent of the input
        maps untouched and the other `1 - mask_p` percent surround-modulated.
        Arguments
        - input: An activation tensor map of shape:
          `[batch, channel, height, width]`.
        """
        # FIX: Currently inefficient random sampling of channels.
        channel_size = input.shape[1]
        # Spread the kernel over the batch and channel dimensions of the input.
        kernel = self.kernel.repeat(channel_size, channel_size, 1, 1)
        # Use the dirac delta (convolutional identity) kernel to mask the sm
        # kernel so as to avoid convolving certain input activations.
        mask_size = math.floor(self.mask_p * channel_size)
        i = torch.randint(low=0, high=channel_size, size=(mask_size, ))
        kernel[i] = nn.init.dirac_(kernel[i])
        return F.conv2d(input, kernel, padding=self.same)