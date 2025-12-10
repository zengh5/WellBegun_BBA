import numpy as np
import torch
import copy
import torch.nn as nn
import math
from torch.nn.parameter import Parameter


def clip_image_values(x, minv, maxv):

    x = torch.max(x, minv)
    x = torch.min(x, maxv)
    return x


def get_label(x):
    s = x.split(' ')
    label = ''
    for l in range(1, len(s)):
        label += s[l] + ' '

    return label


def nnz_pixels(arr):
    return np.count_nonzero(np.sum(np.absolute(arr), axis=0))


class LFAA_GaussianFilter(nn.Module):
    """LFAA Gaussian low-pass filter for frequency decomposition"""

    def __init__(self, kernel_size=17, sigma=4, channels=3):
        super(LFAA_GaussianFilter, self).__init__()
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2
        self.channels = channels

        # Create Gaussian kernel (same as in LFAA paper: k=4 -> 17x17 kernel)
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        mean = (kernel_size - 1) / 2.
        variance = sigma ** 2.
        gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                          torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

        self.gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                         kernel_size=kernel_size, groups=channels,
                                         padding=self.pad, bias=False)
        self.gaussian_filter.weight.data = gaussian_kernel
        self.gaussian_filter.weight.requires_grad = False

    def forward(self, x):
        return self.gaussian_filter(x)


def load_ckpt(model, filename):
    robustmodel = torch.load(filename)['model']
    model.conv1.weight = Parameter(robustmodel['module.model.conv1.weight'])
    model.bn1.weight = Parameter(robustmodel['module.model.bn1.weight'])
    model.bn1.bias = Parameter(robustmodel['module.model.bn1.bias'])
    model.bn1.running_mean = Parameter(robustmodel['module.model.bn1.running_mean'], requires_grad=False)
    model.bn1.running_var = Parameter(robustmodel['module.model.bn1.running_var'], requires_grad=False)
    # model.bn1.num_batches_tracked = Parameter(robustmodel['module.model.bn1.num_batches_tracked'])

    ##### 1.0    24
    model.layer1[0].conv1.weight = Parameter(robustmodel['module.model.layer1.0.conv1.weight'])
    model.layer1[0].bn1.weight = Parameter(robustmodel['module.model.layer1.0.bn1.weight'])
    model.layer1[0].bn1.bias = Parameter(robustmodel['module.model.layer1.0.bn1.bias'])
    model.layer1[0].bn1.running_mean = Parameter(robustmodel['module.model.layer1.0.bn1.running_mean'], requires_grad=False)
    model.layer1[0].bn1.running_var = Parameter(robustmodel['module.model.layer1.0.bn1.running_var'], requires_grad=False)
    # model.layer1[0].bn1.num_batches_tracked = Parameter(robustmodel['module.model.layer1.0.bn1.num_batches_tracked'])

    model.layer1[0].conv2.weight = Parameter(robustmodel['module.model.layer1.0.conv2.weight'])
    model.layer1[0].bn2.weight = Parameter(robustmodel['module.model.layer1.0.bn2.weight'])
    model.layer1[0].bn2.bias = Parameter(robustmodel['module.model.layer1.0.bn2.bias'])
    model.layer1[0].bn2.running_mean = Parameter(robustmodel['module.model.layer1.0.bn2.running_mean'], requires_grad=False)
    model.layer1[0].bn2.running_var = Parameter(robustmodel['module.model.layer1.0.bn2.running_var'], requires_grad=False)
    # model.layer1[0].bn2.num_batches_tracked = Parameter(robustmodel['module.model.layer1.0.bn2.num_batches_tracked'])

    model.layer1[0].conv3.weight = Parameter(robustmodel['module.model.layer1.0.conv3.weight'])
    model.layer1[0].bn3.weight = Parameter(robustmodel['module.model.layer1.0.bn3.weight'])
    model.layer1[0].bn3.bias = Parameter(robustmodel['module.model.layer1.0.bn3.bias'])
    model.layer1[0].bn3.running_mean = Parameter(robustmodel['module.model.layer1.0.bn3.running_mean'], requires_grad=False)
    model.layer1[0].bn3.running_var = Parameter(robustmodel['module.model.layer1.0.bn3.running_var'], requires_grad=False)
    # model.layer1[0].bn3.num_batches_tracked = Parameter(robustmodel['module.model.layer1.0.bn3.num_batches_tracked'])

    model.layer1[0].downsample[0].weight = Parameter(robustmodel['module.model.layer1.0.downsample.0.weight'])
    model.layer1[0].downsample[1].weight = Parameter(robustmodel['module.model.layer1.0.downsample.1.weight'])
    model.layer1[0].downsample[1].bias = Parameter(robustmodel['module.model.layer1.0.downsample.1.bias'])
    model.layer1[0].downsample[1].running_mean = Parameter(robustmodel['module.model.layer1.0.downsample.1.running_mean'], requires_grad=False)
    model.layer1[0].downsample[1].running_var = Parameter(robustmodel['module.model.layer1.0.downsample.1.running_var'], requires_grad=False)
    # model.layer1[0].downsample[1].num_batches_tracked = Parameter(robustmodel['module.model.layer1.0.downsample.1.num_batches_tracked'])

    ##### 1.1    18
    model.layer1[1].conv1.weight = Parameter(robustmodel['module.model.layer1.1.conv1.weight'])
    model.layer1[1].bn1.weight = Parameter(robustmodel['module.model.layer1.1.bn1.weight'])
    model.layer1[1].bn1.bias = Parameter(robustmodel['module.model.layer1.1.bn1.bias'])
    model.layer1[1].bn1.running_mean = Parameter(robustmodel['module.model.layer1.1.bn1.running_mean'], requires_grad=False)
    model.layer1[1].bn1.running_var = Parameter(robustmodel['module.model.layer1.1.bn1.running_var'], requires_grad=False)
    # model.layer1[1].bn1.num_batches_tracked = Parameter(robustmodel['module.model.layer1.1.bn1.num_batches_tracked'])

    model.layer1[1].conv2.weight = Parameter(robustmodel['module.model.layer1.1.conv2.weight'])
    model.layer1[1].bn2.weight = Parameter(robustmodel['module.model.layer1.1.bn2.weight'])
    model.layer1[1].bn2.bias = Parameter(robustmodel['module.model.layer1.1.bn2.bias'])
    model.layer1[1].bn2.running_mean = Parameter(robustmodel['module.model.layer1.1.bn2.running_mean'], requires_grad=False)
    model.layer1[1].bn2.running_var = Parameter(robustmodel['module.model.layer1.1.bn2.running_var'], requires_grad=False)
    # model.layer1[1].bn2.num_batches_tracked = Parameter(robustmodel['module.model.layer1.1.bn2.num_batches_tracked'])

    model.layer1[1].conv3.weight = Parameter(robustmodel['module.model.layer1.1.conv3.weight'])
    model.layer1[1].bn3.weight = Parameter(robustmodel['module.model.layer1.1.bn3.weight'])
    model.layer1[1].bn3.bias = Parameter(robustmodel['module.model.layer1.1.bn3.bias'])
    model.layer1[1].bn3.running_mean = Parameter(robustmodel['module.model.layer1.1.bn3.running_mean'], requires_grad=False)
    model.layer1[1].bn3.running_var = Parameter(robustmodel['module.model.layer1.1.bn3.running_var'], requires_grad=False)
    # model.layer1[1].bn3.num_batches_tracked = Parameter(robustmodel['module.model.layer1.1.bn3.num_batches_tracked'])

    ##### 1.2  18
    model.layer1[2].conv1.weight = Parameter(robustmodel['module.model.layer1.2.conv1.weight'])
    model.layer1[2].bn1.weight = Parameter(robustmodel['module.model.layer1.2.bn1.weight'])
    model.layer1[2].bn1.bias = Parameter(robustmodel['module.model.layer1.2.bn1.bias'])
    model.layer1[2].bn1.running_mean = Parameter(robustmodel['module.model.layer1.2.bn1.running_mean'], requires_grad=False)
    model.layer1[2].bn1.running_var = Parameter(robustmodel['module.model.layer1.2.bn1.running_var'], requires_grad=False)
    # model.layer1[2].bn1.num_batches_tracked = Parameter(robustmodel['module.model.layer1.2.bn1.num_batches_tracked'])

    model.layer1[2].conv2.weight = Parameter(robustmodel['module.model.layer1.2.conv2.weight'])
    model.layer1[2].bn2.weight = Parameter(robustmodel['module.model.layer1.2.bn2.weight'])
    model.layer1[2].bn2.bias = Parameter(robustmodel['module.model.layer1.2.bn2.bias'])
    model.layer1[2].bn2.running_mean = Parameter(robustmodel['module.model.layer1.2.bn2.running_mean'], requires_grad=False)
    model.layer1[2].bn2.running_var = Parameter(robustmodel['module.model.layer1.2.bn2.running_var'], requires_grad=False)
    # model.layer1[2].bn2.num_batches_tracked = Parameter(robustmodel['module.model.layer1.2.bn2.num_batches_tracked'])

    model.layer1[2].conv3.weight = Parameter(robustmodel['module.model.layer1.2.conv3.weight'])
    model.layer1[2].bn3.weight = Parameter(robustmodel['module.model.layer1.2.bn3.weight'])
    model.layer1[2].bn3.bias = Parameter(robustmodel['module.model.layer1.2.bn3.bias'])
    model.layer1[2].bn3.running_mean = Parameter(robustmodel['module.model.layer1.2.bn3.running_mean'], requires_grad=False)
    model.layer1[2].bn3.running_var = Parameter(robustmodel['module.model.layer1.2.bn3.running_var'], requires_grad=False)
    # model.layer1[2].bn3.num_batches_tracked = Parameter(robustmodel['module.model.layer1.2.bn3.num_batches_tracked'])

    ##### 2.0    24
    model.layer2[0].conv1.weight = Parameter(robustmodel['module.model.layer2.0.conv1.weight'])
    model.layer2[0].bn1.weight = Parameter(robustmodel['module.model.layer2.0.bn1.weight'])
    model.layer2[0].bn1.bias = Parameter(robustmodel['module.model.layer2.0.bn1.bias'])
    model.layer2[0].bn1.running_mean = Parameter(robustmodel['module.model.layer2.0.bn1.running_mean'], requires_grad=False)
    model.layer2[0].bn1.running_var = Parameter(robustmodel['module.model.layer2.0.bn1.running_var'], requires_grad=False)
    # model.layer2[0].bn1.num_batches_tracked = Parameter(robustmodel['module.model.layer2.0.bn1.num_batches_tracked'])

    model.layer2[0].conv2.weight = Parameter(robustmodel['module.model.layer2.0.conv2.weight'])
    model.layer2[0].bn2.weight = Parameter(robustmodel['module.model.layer2.0.bn2.weight'])
    model.layer2[0].bn2.bias = Parameter(robustmodel['module.model.layer2.0.bn2.bias'])
    model.layer2[0].bn2.running_mean = Parameter(robustmodel['module.model.layer2.0.bn2.running_mean'], requires_grad=False)
    model.layer2[0].bn2.running_var = Parameter(robustmodel['module.model.layer2.0.bn2.running_var'], requires_grad=False)
    # model.layer2[0].bn2.num_batches_tracked = Parameter(robustmodel['module.model.layer2.0.bn2.num_batches_tracked'])

    model.layer2[0].conv3.weight = Parameter(robustmodel['module.model.layer2.0.conv3.weight'])
    model.layer2[0].bn3.weight = Parameter(robustmodel['module.model.layer2.0.bn3.weight'])
    model.layer2[0].bn3.bias = Parameter(robustmodel['module.model.layer2.0.bn3.bias'])
    model.layer2[0].bn3.running_mean = Parameter(robustmodel['module.model.layer2.0.bn3.running_mean'], requires_grad=False)
    model.layer2[0].bn3.running_var = Parameter(robustmodel['module.model.layer2.0.bn3.running_var'], requires_grad=False)
    # model.layer2[0].bn3.num_batches_tracked = Parameter(robustmodel['module.model.layer2.0.bn3.num_batches_tracked'])

    model.layer2[0].downsample[0].weight = Parameter(robustmodel['module.model.layer2.0.downsample.0.weight'])
    model.layer2[0].downsample[1].weight = Parameter(robustmodel['module.model.layer2.0.downsample.1.weight'])
    model.layer2[0].downsample[1].bias = Parameter(robustmodel['module.model.layer2.0.downsample.1.bias'])
    model.layer2[0].downsample[1].running_mean = Parameter(
        robustmodel['module.model.layer2.0.downsample.1.running_mean'], requires_grad=False)
    model.layer2[0].downsample[1].running_var = Parameter(robustmodel['module.model.layer2.0.downsample.1.running_var'], requires_grad=False)
    # model.layer2[0].downsample[1].num_batches_tracked = Parameter(
    #     robustmodel['module.model.layer2.0.downsample.1.num_batches_tracked'])

    ##### 2.1    18
    model.layer2[1].conv1.weight = Parameter(robustmodel['module.model.layer2.1.conv1.weight'])
    model.layer2[1].bn1.weight = Parameter(robustmodel['module.model.layer2.1.bn1.weight'])
    model.layer2[1].bn1.bias = Parameter(robustmodel['module.model.layer2.1.bn1.bias'])
    model.layer2[1].bn1.running_mean = Parameter(robustmodel['module.model.layer2.1.bn1.running_mean'], requires_grad=False)
    model.layer2[1].bn1.running_var = Parameter(robustmodel['module.model.layer2.1.bn1.running_var'], requires_grad=False)
    # model.layer2[1].bn1.num_batches_tracked = Parameter(robustmodel['module.model.layer2.1.bn1.num_batches_tracked'])

    model.layer2[1].conv2.weight = Parameter(robustmodel['module.model.layer2.1.conv2.weight'])
    model.layer2[1].bn2.weight = Parameter(robustmodel['module.model.layer2.1.bn2.weight'])
    model.layer2[1].bn2.bias = Parameter(robustmodel['module.model.layer2.1.bn2.bias'])
    model.layer2[1].bn2.running_mean = Parameter(robustmodel['module.model.layer2.1.bn2.running_mean'], requires_grad=False)
    model.layer2[1].bn2.running_var = Parameter(robustmodel['module.model.layer2.1.bn2.running_var'], requires_grad=False)
    # model.layer2[1].bn2.num_batches_tracked = Parameter(robustmodel['module.model.layer2.1.bn2.num_batches_tracked'])

    model.layer2[1].conv3.weight = Parameter(robustmodel['module.model.layer2.1.conv3.weight'])
    model.layer2[1].bn3.weight = Parameter(robustmodel['module.model.layer2.1.bn3.weight'])
    model.layer2[1].bn3.bias = Parameter(robustmodel['module.model.layer2.1.bn3.bias'])
    model.layer2[1].bn3.running_mean = Parameter(robustmodel['module.model.layer2.1.bn3.running_mean'], requires_grad=False)
    model.layer2[1].bn3.running_var = Parameter(robustmodel['module.model.layer2.1.bn3.running_var'], requires_grad=False)
    # model.layer2[1].bn3.num_batches_tracked = Parameter(robustmodel['module.model.layer2.1.bn3.num_batches_tracked'])

    ##### 2.2  18
    model.layer2[2].conv1.weight = Parameter(robustmodel['module.model.layer2.2.conv1.weight'])
    model.layer2[2].bn1.weight = Parameter(robustmodel['module.model.layer2.2.bn1.weight'])
    model.layer2[2].bn1.bias = Parameter(robustmodel['module.model.layer2.2.bn1.bias'])
    model.layer2[2].bn1.running_mean = Parameter(robustmodel['module.model.layer2.2.bn1.running_mean'], requires_grad=False)
    model.layer2[2].bn1.running_var = Parameter(robustmodel['module.model.layer2.2.bn1.running_var'], requires_grad=False)
    # model.layer2[2].bn1.num_batches_tracked = Parameter(robustmodel['module.model.layer2.2.bn1.num_batches_tracked'])

    model.layer2[2].conv2.weight = Parameter(robustmodel['module.model.layer2.2.conv2.weight'])
    model.layer2[2].bn2.weight = Parameter(robustmodel['module.model.layer2.2.bn2.weight'])
    model.layer2[2].bn2.bias = Parameter(robustmodel['module.model.layer2.2.bn2.bias'])
    model.layer2[2].bn2.running_mean = Parameter(robustmodel['module.model.layer2.2.bn2.running_mean'], requires_grad=False)
    model.layer2[2].bn2.running_var = Parameter(robustmodel['module.model.layer2.2.bn2.running_var'], requires_grad=False)
    # model.layer2[2].bn2.num_batches_tracked = Parameter(robustmodel['module.model.layer2.2.bn2.num_batches_tracked'])

    model.layer2[2].conv3.weight = Parameter(robustmodel['module.model.layer2.2.conv3.weight'])
    model.layer2[2].bn3.weight = Parameter(robustmodel['module.model.layer2.2.bn3.weight'])
    model.layer2[2].bn3.bias = Parameter(robustmodel['module.model.layer2.2.bn3.bias'])
    model.layer2[2].bn3.running_mean = Parameter(robustmodel['module.model.layer2.2.bn3.running_mean'], requires_grad=False)
    model.layer2[2].bn3.running_var = Parameter(robustmodel['module.model.layer2.2.bn3.running_var'], requires_grad=False)
    # model.layer2[2].bn3.num_batches_tracked = Parameter(robustmodel['module.model.layer2.2.bn3.num_batches_tracked'])

    ##### 2.3  18
    model.layer2[3].conv1.weight = Parameter(robustmodel['module.model.layer2.3.conv1.weight'])
    model.layer2[3].bn1.weight = Parameter(robustmodel['module.model.layer2.3.bn1.weight'])
    model.layer2[3].bn1.bias = Parameter(robustmodel['module.model.layer2.3.bn1.bias'])
    model.layer2[3].bn1.running_mean = Parameter(robustmodel['module.model.layer2.3.bn1.running_mean'], requires_grad=False)
    model.layer2[3].bn1.running_var = Parameter(robustmodel['module.model.layer2.3.bn1.running_var'], requires_grad=False)
    # model.layer2[3].bn1.num_batches_tracked = Parameter(robustmodel['module.model.layer2.3.bn1.num_batches_tracked'])

    model.layer2[3].conv2.weight = Parameter(robustmodel['module.model.layer2.3.conv2.weight'])
    model.layer2[3].bn2.weight = Parameter(robustmodel['module.model.layer2.3.bn2.weight'])
    model.layer2[3].bn2.bias = Parameter(robustmodel['module.model.layer2.3.bn2.bias'])
    model.layer2[3].bn2.running_mean = Parameter(robustmodel['module.model.layer2.3.bn2.running_mean'], requires_grad=False)
    model.layer2[3].bn2.running_var = Parameter(robustmodel['module.model.layer2.3.bn2.running_var'], requires_grad=False)
    # model.layer2[3].bn2.num_batches_tracked = Parameter(robustmodel['module.model.layer2.3.bn2.num_batches_tracked'])

    model.layer2[3].conv3.weight = Parameter(robustmodel['module.model.layer2.3.conv3.weight'])
    model.layer2[3].bn3.weight = Parameter(robustmodel['module.model.layer2.3.bn3.weight'])
    model.layer2[3].bn3.bias = Parameter(robustmodel['module.model.layer2.3.bn3.bias'])
    model.layer2[3].bn3.running_mean = Parameter(robustmodel['module.model.layer2.3.bn3.running_mean'], requires_grad=False)
    model.layer2[3].bn3.running_var = Parameter(robustmodel['module.model.layer2.3.bn3.running_var'], requires_grad=False)
    # model.layer2[3].bn3.num_batches_tracked = Parameter(robustmodel['module.model.layer2.3.bn3.num_batches_tracked'])

    ##### 3.0    24
    model.layer3[0].conv1.weight = Parameter(robustmodel['module.model.layer3.0.conv1.weight'])
    model.layer3[0].bn1.weight = Parameter(robustmodel['module.model.layer3.0.bn1.weight'])
    model.layer3[0].bn1.bias = Parameter(robustmodel['module.model.layer3.0.bn1.bias'])
    model.layer3[0].bn1.running_mean = Parameter(robustmodel['module.model.layer3.0.bn1.running_mean'], requires_grad=False)
    model.layer3[0].bn1.running_var = Parameter(robustmodel['module.model.layer3.0.bn1.running_var'], requires_grad=False)
    # model.layer3[0].bn1.num_batches_tracked = Parameter(robustmodel['module.model.layer3.0.bn1.num_batches_tracked'])

    model.layer3[0].conv2.weight = Parameter(robustmodel['module.model.layer3.0.conv2.weight'])
    model.layer3[0].bn2.weight = Parameter(robustmodel['module.model.layer3.0.bn2.weight'])
    model.layer3[0].bn2.bias = Parameter(robustmodel['module.model.layer3.0.bn2.bias'])
    model.layer3[0].bn2.running_mean = Parameter(robustmodel['module.model.layer3.0.bn2.running_mean'], requires_grad=False)
    model.layer3[0].bn2.running_var = Parameter(robustmodel['module.model.layer3.0.bn2.running_var'], requires_grad=False)
    # model.layer3[0].bn2.num_batches_tracked = Parameter(robustmodel['module.model.layer3.0.bn2.num_batches_tracked'])

    model.layer3[0].conv3.weight = Parameter(robustmodel['module.model.layer3.0.conv3.weight'])
    model.layer3[0].bn3.weight = Parameter(robustmodel['module.model.layer3.0.bn3.weight'])
    model.layer3[0].bn3.bias = Parameter(robustmodel['module.model.layer3.0.bn3.bias'])
    model.layer3[0].bn3.running_mean = Parameter(robustmodel['module.model.layer3.0.bn3.running_mean'], requires_grad=False)
    model.layer3[0].bn3.running_var = Parameter(robustmodel['module.model.layer3.0.bn3.running_var'], requires_grad=False)
    # model.layer3[0].bn3.num_batches_tracked = Parameter(robustmodel['module.model.layer3.0.bn3.num_batches_tracked'])

    model.layer3[0].downsample[0].weight = Parameter(robustmodel['module.model.layer3.0.downsample.0.weight'])
    model.layer3[0].downsample[1].weight = Parameter(robustmodel['module.model.layer3.0.downsample.1.weight'])
    model.layer3[0].downsample[1].bias = Parameter(robustmodel['module.model.layer3.0.downsample.1.bias'])
    model.layer3[0].downsample[1].running_mean = Parameter(
        robustmodel['module.model.layer3.0.downsample.1.running_mean'], requires_grad=False)
    model.layer3[0].downsample[1].running_var = Parameter(robustmodel['module.model.layer3.0.downsample.1.running_var'], requires_grad=False)
    # model.layer3[0].downsample[1].num_batches_tracked = Parameter(
    #     robustmodel['module.model.layer3.0.downsample.1.num_batches_tracked'])

    ##### 3.1    18
    model.layer3[1].conv1.weight = Parameter(robustmodel['module.model.layer3.1.conv1.weight'])
    model.layer3[1].bn1.weight = Parameter(robustmodel['module.model.layer3.1.bn1.weight'])
    model.layer3[1].bn1.bias = Parameter(robustmodel['module.model.layer3.1.bn1.bias'])
    model.layer3[1].bn1.running_mean = Parameter(robustmodel['module.model.layer3.1.bn1.running_mean'], requires_grad=False)
    model.layer3[1].bn1.running_var = Parameter(robustmodel['module.model.layer3.1.bn1.running_var'], requires_grad=False)
    # model.layer3[1].bn1.num_batches_tracked = Parameter(robustmodel['module.model.layer3.1.bn1.num_batches_tracked'])

    model.layer3[1].conv2.weight = Parameter(robustmodel['module.model.layer3.1.conv2.weight'])
    model.layer3[1].bn2.weight = Parameter(robustmodel['module.model.layer3.1.bn2.weight'])
    model.layer3[1].bn2.bias = Parameter(robustmodel['module.model.layer3.1.bn2.bias'])
    model.layer3[1].bn2.running_mean = Parameter(robustmodel['module.model.layer3.1.bn2.running_mean'], requires_grad=False)
    model.layer3[1].bn2.running_var = Parameter(robustmodel['module.model.layer3.1.bn2.running_var'], requires_grad=False)
    # model.layer3[1].bn2.num_batches_tracked = Parameter(robustmodel['module.model.layer3.1.bn2.num_batches_tracked'])

    model.layer3[1].conv3.weight = Parameter(robustmodel['module.model.layer3.1.conv3.weight'])
    model.layer3[1].bn3.weight = Parameter(robustmodel['module.model.layer3.1.bn3.weight'])
    model.layer3[1].bn3.bias = Parameter(robustmodel['module.model.layer3.1.bn3.bias'])
    model.layer3[1].bn3.running_mean = Parameter(robustmodel['module.model.layer3.1.bn3.running_mean'], requires_grad=False)
    model.layer3[1].bn3.running_var = Parameter(robustmodel['module.model.layer3.1.bn3.running_var'], requires_grad=False)
    # model.layer3[1].bn3.num_batches_tracked = Parameter(robustmodel['module.model.layer3.1.bn3.num_batches_tracked'])

    ##### 3.2  18
    model.layer3[2].conv1.weight = Parameter(robustmodel['module.model.layer3.2.conv1.weight'])
    model.layer3[2].bn1.weight = Parameter(robustmodel['module.model.layer3.2.bn1.weight'])
    model.layer3[2].bn1.bias = Parameter(robustmodel['module.model.layer3.2.bn1.bias'])
    model.layer3[2].bn1.running_mean = Parameter(robustmodel['module.model.layer3.2.bn1.running_mean'], requires_grad=False)
    model.layer3[2].bn1.running_var = Parameter(robustmodel['module.model.layer3.2.bn1.running_var'], requires_grad=False)
    # model.layer3[2].bn1.num_batches_tracked = Parameter(robustmodel['module.model.layer3.2.bn1.num_batches_tracked'])

    model.layer3[2].conv2.weight = Parameter(robustmodel['module.model.layer3.2.conv2.weight'])
    model.layer3[2].bn2.weight = Parameter(robustmodel['module.model.layer3.2.bn2.weight'])
    model.layer3[2].bn2.bias = Parameter(robustmodel['module.model.layer3.2.bn2.bias'])
    model.layer3[2].bn2.running_mean = Parameter(robustmodel['module.model.layer3.2.bn2.running_mean'], requires_grad=False)
    model.layer3[2].bn2.running_var = Parameter(robustmodel['module.model.layer3.2.bn2.running_var'], requires_grad=False)
    # model.layer3[2].bn2.num_batches_tracked = Parameter(robustmodel['module.model.layer3.2.bn2.num_batches_tracked'])

    model.layer3[2].conv3.weight = Parameter(robustmodel['module.model.layer3.2.conv3.weight'])
    model.layer3[2].bn3.weight = Parameter(robustmodel['module.model.layer3.2.bn3.weight'])
    model.layer3[2].bn3.bias = Parameter(robustmodel['module.model.layer3.2.bn3.bias'])
    model.layer3[2].bn3.running_mean = Parameter(robustmodel['module.model.layer3.2.bn3.running_mean'], requires_grad=False)
    model.layer3[2].bn3.running_var = Parameter(robustmodel['module.model.layer3.2.bn3.running_var'], requires_grad=False)
    # model.layer3[2].bn3.num_batches_tracked = Parameter(robustmodel['module.model.layer3.2.bn3.num_batches_tracked'])

    ##### 3.3  18
    model.layer3[3].conv1.weight = Parameter(robustmodel['module.model.layer3.3.conv1.weight'])
    model.layer3[3].bn1.weight = Parameter(robustmodel['module.model.layer3.3.bn1.weight'])
    model.layer3[3].bn1.bias = Parameter(robustmodel['module.model.layer3.3.bn1.bias'])
    model.layer3[3].bn1.running_mean = Parameter(robustmodel['module.model.layer3.3.bn1.running_mean'], requires_grad=False)
    model.layer3[3].bn1.running_var = Parameter(robustmodel['module.model.layer3.3.bn1.running_var'], requires_grad=False)
    # model.layer3[3].bn1.num_batches_tracked = Parameter(robustmodel['module.model.layer3.3.bn1.num_batches_tracked'])

    model.layer3[3].conv2.weight = Parameter(robustmodel['module.model.layer3.3.conv2.weight'])
    model.layer3[3].bn2.weight = Parameter(robustmodel['module.model.layer3.3.bn2.weight'])
    model.layer3[3].bn2.bias = Parameter(robustmodel['module.model.layer3.3.bn2.bias'])
    model.layer3[3].bn2.running_mean = Parameter(robustmodel['module.model.layer3.3.bn2.running_mean'], requires_grad=False)
    model.layer3[3].bn2.running_var = Parameter(robustmodel['module.model.layer3.3.bn2.running_var'], requires_grad=False)
    # model.layer3[3].bn2.num_batches_tracked = Parameter(robustmodel['module.model.layer3.3.bn2.num_batches_tracked'])

    model.layer3[3].conv3.weight = Parameter(robustmodel['module.model.layer3.3.conv3.weight'])
    model.layer3[3].bn3.weight = Parameter(robustmodel['module.model.layer3.3.bn3.weight'])
    model.layer3[3].bn3.bias = Parameter(robustmodel['module.model.layer3.3.bn3.bias'])
    model.layer3[3].bn3.running_mean = Parameter(robustmodel['module.model.layer3.3.bn3.running_mean'], requires_grad=False)
    model.layer3[3].bn3.running_var = Parameter(robustmodel['module.model.layer3.3.bn3.running_var'], requires_grad=False)
    # model.layer3[3].bn3.num_batches_tracked = Parameter(robustmodel['module.model.layer3.3.bn3.num_batches_tracked'])

    ##### 3.4  18
    model.layer3[4].conv1.weight = Parameter(robustmodel['module.model.layer3.4.conv1.weight'])
    model.layer3[4].bn1.weight = Parameter(robustmodel['module.model.layer3.4.bn1.weight'])
    model.layer3[4].bn1.bias = Parameter(robustmodel['module.model.layer3.4.bn1.bias'])
    model.layer3[4].bn1.running_mean = Parameter(robustmodel['module.model.layer3.4.bn1.running_mean'], requires_grad=False)
    model.layer3[4].bn1.running_var = Parameter(robustmodel['module.model.layer3.4.bn1.running_var'], requires_grad=False)
    # model.layer3[4].bn1.num_batches_tracked = Parameter(robustmodel['module.model.layer3.4.bn1.num_batches_tracked'])

    model.layer3[4].conv2.weight = Parameter(robustmodel['module.model.layer3.4.conv2.weight'])
    model.layer3[4].bn2.weight = Parameter(robustmodel['module.model.layer3.4.bn2.weight'])
    model.layer3[4].bn2.bias = Parameter(robustmodel['module.model.layer3.4.bn2.bias'])
    model.layer3[4].bn2.running_mean = Parameter(robustmodel['module.model.layer3.4.bn2.running_mean'], requires_grad=False)
    model.layer3[4].bn2.running_var = Parameter(robustmodel['module.model.layer3.4.bn2.running_var'], requires_grad=False)
    # model.layer3[4].bn2.num_batches_tracked = Parameter(robustmodel['module.model.layer3.4.bn2.num_batches_tracked'])

    model.layer3[4].conv3.weight = Parameter(robustmodel['module.model.layer3.4.conv3.weight'])
    model.layer3[4].bn3.weight = Parameter(robustmodel['module.model.layer3.4.bn3.weight'])
    model.layer3[4].bn3.bias = Parameter(robustmodel['module.model.layer3.4.bn3.bias'])
    model.layer3[4].bn3.running_mean = Parameter(robustmodel['module.model.layer3.4.bn3.running_mean'], requires_grad=False)
    model.layer3[4].bn3.running_var = Parameter(robustmodel['module.model.layer3.4.bn3.running_var'], requires_grad=False)
    # model.layer3[4].bn3.num_batches_tracked = Parameter(robustmodel['module.model.layer3.4.bn3.num_batches_tracked'])

    ##### 3.5  18
    model.layer3[5].conv1.weight = Parameter(robustmodel['module.model.layer3.5.conv1.weight'])
    model.layer3[5].bn1.weight = Parameter(robustmodel['module.model.layer3.5.bn1.weight'])
    model.layer3[5].bn1.bias = Parameter(robustmodel['module.model.layer3.5.bn1.bias'])
    model.layer3[5].bn1.running_mean = Parameter(robustmodel['module.model.layer3.5.bn1.running_mean'], requires_grad=False)
    model.layer3[5].bn1.running_var = Parameter(robustmodel['module.model.layer3.5.bn1.running_var'], requires_grad=False)
    # model.layer3[5].bn1.num_batches_tracked = Parameter(robustmodel['module.model.layer3.5.bn1.num_batches_tracked'])

    model.layer3[5].conv2.weight = Parameter(robustmodel['module.model.layer3.5.conv2.weight'])
    model.layer3[5].bn2.weight = Parameter(robustmodel['module.model.layer3.5.bn2.weight'])
    model.layer3[5].bn2.bias = Parameter(robustmodel['module.model.layer3.5.bn2.bias'])
    model.layer3[5].bn2.running_mean = Parameter(robustmodel['module.model.layer3.5.bn2.running_mean'], requires_grad=False)
    model.layer3[5].bn2.running_var = Parameter(robustmodel['module.model.layer3.5.bn2.running_var'], requires_grad=False)
    # model.layer3[5].bn2.num_batches_tracked = Parameter(robustmodel['module.model.layer3.5.bn2.num_batches_tracked'])

    model.layer3[5].conv3.weight = Parameter(robustmodel['module.model.layer3.5.conv3.weight'])
    model.layer3[5].bn3.weight = Parameter(robustmodel['module.model.layer3.5.bn3.weight'])
    model.layer3[5].bn3.bias = Parameter(robustmodel['module.model.layer3.5.bn3.bias'])
    model.layer3[5].bn3.running_mean = Parameter(robustmodel['module.model.layer3.5.bn3.running_mean'], requires_grad=False)
    model.layer3[5].bn3.running_var = Parameter(robustmodel['module.model.layer3.5.bn3.running_var'], requires_grad=False)
    # model.layer3[5].bn3.num_batches_tracked = Parameter(robustmodel['module.model.layer3.5.bn3.num_batches_tracked'])

    ##### 4.0    24
    model.layer4[0].conv1.weight = Parameter(robustmodel['module.model.layer4.0.conv1.weight'])
    model.layer4[0].bn1.weight = Parameter(robustmodel['module.model.layer4.0.bn1.weight'])
    model.layer4[0].bn1.bias = Parameter(robustmodel['module.model.layer4.0.bn1.bias'])
    model.layer4[0].bn1.running_mean = Parameter(robustmodel['module.model.layer4.0.bn1.running_mean'], requires_grad=False)
    model.layer4[0].bn1.running_var = Parameter(robustmodel['module.model.layer4.0.bn1.running_var'], requires_grad=False)
    # model.layer4[0].bn1.num_batches_tracked = Parameter(robustmodel['module.model.layer4.0.bn1.num_batches_tracked'])

    model.layer4[0].conv2.weight = Parameter(robustmodel['module.model.layer4.0.conv2.weight'])
    model.layer4[0].bn2.weight = Parameter(robustmodel['module.model.layer4.0.bn2.weight'])
    model.layer4[0].bn2.bias = Parameter(robustmodel['module.model.layer4.0.bn2.bias'])
    model.layer4[0].bn2.running_mean = Parameter(robustmodel['module.model.layer4.0.bn2.running_mean'], requires_grad=False)
    model.layer4[0].bn2.running_var = Parameter(robustmodel['module.model.layer4.0.bn2.running_var'], requires_grad=False)
    # model.layer4[0].bn2.num_batches_tracked = Parameter(robustmodel['module.model.layer4.0.bn2.num_batches_tracked'])

    model.layer4[0].conv3.weight = Parameter(robustmodel['module.model.layer4.0.conv3.weight'])
    model.layer4[0].bn3.weight = Parameter(robustmodel['module.model.layer4.0.bn3.weight'])
    model.layer4[0].bn3.bias = Parameter(robustmodel['module.model.layer4.0.bn3.bias'])
    model.layer4[0].bn3.running_mean = Parameter(robustmodel['module.model.layer4.0.bn3.running_mean'], requires_grad=False)
    model.layer4[0].bn3.running_var = Parameter(robustmodel['module.model.layer4.0.bn3.running_var'], requires_grad=False)
    # model.layer4[0].bn3.num_batches_tracked = Parameter(robustmodel['module.model.layer4.0.bn3.num_batches_tracked'])

    model.layer4[0].downsample[0].weight = Parameter(robustmodel['module.model.layer4.0.downsample.0.weight'])
    model.layer4[0].downsample[1].weight = Parameter(robustmodel['module.model.layer4.0.downsample.1.weight'])
    model.layer4[0].downsample[1].bias = Parameter(robustmodel['module.model.layer4.0.downsample.1.bias'])
    model.layer4[0].downsample[1].running_mean = Parameter(
        robustmodel['module.model.layer4.0.downsample.1.running_mean'], requires_grad=False)
    model.layer4[0].downsample[1].running_var = Parameter(robustmodel['module.model.layer4.0.downsample.1.running_var'], requires_grad=False)
    # model.layer4[0].downsample[1].num_batches_tracked = Parameter(
    #     robustmodel['module.model.layer4.0.downsample.1.num_batches_tracked'])

    ##### 4.1    18
    model.layer4[1].conv1.weight = Parameter(robustmodel['module.model.layer4.1.conv1.weight'])
    model.layer4[1].bn1.weight = Parameter(robustmodel['module.model.layer4.1.bn1.weight'])
    model.layer4[1].bn1.bias = Parameter(robustmodel['module.model.layer4.1.bn1.bias'])
    model.layer4[1].bn1.running_mean = Parameter(robustmodel['module.model.layer4.1.bn1.running_mean'], requires_grad=False)
    model.layer4[1].bn1.running_var = Parameter(robustmodel['module.model.layer4.1.bn1.running_var'], requires_grad=False)
    # model.layer4[1].bn1.num_batches_tracked = Parameter(robustmodel['module.model.layer4.1.bn1.num_batches_tracked'])

    model.layer4[1].conv2.weight = Parameter(robustmodel['module.model.layer4.1.conv2.weight'])
    model.layer4[1].bn2.weight = Parameter(robustmodel['module.model.layer4.1.bn2.weight'])
    model.layer4[1].bn2.bias = Parameter(robustmodel['module.model.layer4.1.bn2.bias'])
    model.layer4[1].bn2.running_mean = Parameter(robustmodel['module.model.layer4.1.bn2.running_mean'], requires_grad=False)
    model.layer4[1].bn2.running_var = Parameter(robustmodel['module.model.layer4.1.bn2.running_var'], requires_grad=False)
    # model.layer4[1].bn2.num_batches_tracked = Parameter(robustmodel['module.model.layer4.1.bn2.num_batches_tracked'])

    model.layer4[1].conv3.weight = Parameter(robustmodel['module.model.layer4.1.conv3.weight'])
    model.layer4[1].bn3.weight = Parameter(robustmodel['module.model.layer4.1.bn3.weight'])
    model.layer4[1].bn3.bias = Parameter(robustmodel['module.model.layer4.1.bn3.bias'])
    model.layer4[1].bn3.running_mean = Parameter(robustmodel['module.model.layer4.1.bn3.running_mean'], requires_grad=False)
    model.layer4[1].bn3.running_var = Parameter(robustmodel['module.model.layer4.1.bn3.running_var'], requires_grad=False)
    # model.layer4[1].bn3.num_batches_tracked = Parameter(robustmodel['module.model.layer4.1.bn3.num_batches_tracked'])

    ##### 4.2  18
    model.layer4[2].conv1.weight = Parameter(robustmodel['module.model.layer4.2.conv1.weight'])
    model.layer4[2].bn1.weight = Parameter(robustmodel['module.model.layer4.2.bn1.weight'])
    model.layer4[2].bn1.bias = Parameter(robustmodel['module.model.layer4.2.bn1.bias'])
    model.layer4[2].bn1.running_mean = Parameter(robustmodel['module.model.layer4.2.bn1.running_mean'], requires_grad=False)
    model.layer4[2].bn1.running_var = Parameter(robustmodel['module.model.layer4.2.bn1.running_var'], requires_grad=False)
    # model.layer4[2].bn1.num_batches_tracked = Parameter(robustmodel['module.model.layer4.2.bn1.num_batches_tracked'])

    model.layer4[2].conv2.weight = Parameter(robustmodel['module.model.layer4.2.conv2.weight'])
    model.layer4[2].bn2.weight = Parameter(robustmodel['module.model.layer4.2.bn2.weight'])
    model.layer4[2].bn2.bias = Parameter(robustmodel['module.model.layer4.2.bn2.bias'])
    model.layer4[2].bn2.running_mean = Parameter(robustmodel['module.model.layer4.2.bn2.running_mean'], requires_grad=False)
    model.layer4[2].bn2.running_var = Parameter(robustmodel['module.model.layer4.2.bn2.running_var'], requires_grad=False)
    # model.layer4[2].bn2.num_batches_tracked = Parameter(robustmodel['module.model.layer4.2.bn2.num_batches_tracked'])

    model.layer4[2].conv3.weight = Parameter(robustmodel['module.model.layer4.2.conv3.weight'])
    model.layer4[2].bn3.weight = Parameter(robustmodel['module.model.layer4.2.bn3.weight'])
    model.layer4[2].bn3.bias = Parameter(robustmodel['module.model.layer4.2.bn3.bias'])
    model.layer4[2].bn3.running_mean = Parameter(robustmodel['module.model.layer4.2.bn3.running_mean'], requires_grad=False)
    model.layer4[2].bn3.running_var = Parameter(robustmodel['module.model.layer4.2.bn3.running_var'], requires_grad=False)
    # model.layer4[2].bn3.num_batches_tracked = Parameter(robustmodel['module.model.layer4.2.bn3.num_batches_tracked'])

    model.fc.weight = Parameter(robustmodel['module.model.fc.weight'])
    model.fc.bias = Parameter(robustmodel['module.model.fc.bias'])
    return model

