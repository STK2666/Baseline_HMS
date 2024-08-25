from click import style
from torch import nn

import torch.nn.functional as F
import torch
from op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d, conv2d_gradfix
# from modules.util import kpt2heatmap
import math


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer("kernel", kernel)

        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)

        return out


class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})"
        )


class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        style_dim,
        demodulate=True,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        fused=True,
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channels * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, kernel_size, kernel_size)
        )

        self.modulation = EqualLinear(style_dim, in_channels, bias_init=1)

        self.demodulate = demodulate
        self.fused = fused

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.in_channels}, {self.out_channels}, {self.kernel_size}, "
            f"upsample={self.upsample}, downsample={self.downsample})"
        )

    def forward(self, input, style):
        batch, in_channels, height, width = input.shape

        if not self.fused:
            weight = self.scale * self.weight.squeeze(0)
            style = self.modulation(style)

            if self.demodulate:
                w = weight.unsqueeze(0) * style.view(batch, 1, in_channels, 1, 1)
                dcoefs = (w.square().sum((2, 3, 4)) + 1e-8).rsqrt()

            input = input * style.reshape(batch, in_channels, 1, 1)

            if self.upsample:
                weight = weight.transpose(0, 1)
                out = conv2d_gradfix.conv_transpose2d(
                    input, weight, padding=0, stride=2
                )
                out = self.blur(out)

            elif self.downsample:
                input = self.blur(input)
                out = conv2d_gradfix.conv2d(input, weight, padding=0, stride=2)

            else:
                out = conv2d_gradfix.conv2d(input, weight, padding=self.padding)

            if self.demodulate:
                out = out * dcoefs.view(batch, -1, 1, 1)

            return out

        style = self.modulation(style).view(batch, 1, in_channels, 1, 1)
        weight = self.scale * self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channels, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channels, in_channels, self.kernel_size, self.kernel_size
        )

        if self.upsample:
            input = input.view(1, batch * in_channels, height, width)
            weight = weight.view(
                batch, self.out_channels, in_channels, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channels, self.out_channels, self.kernel_size, self.kernel_size
            )
            out = conv2d_gradfix.conv_transpose2d(
                input, weight, padding=0, stride=2, groups=batch
            )
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channels, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channels, height, width)
            out = conv2d_gradfix.conv2d(
                input, weight, padding=0, stride=2, groups=batch
            )
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channels, height, width)

        else:
            input = input.view(1, batch * in_channels, height, width)
            out = conv2d_gradfix.conv2d(
                input, weight, padding=self.padding, groups=batch
            )
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channels, height, width)

        return out


class StyledConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        style_dim,
        upsample=False,
        blur_kernel=[1, 3, 3, 1],
        demodulate=True,
    ):
        super().__init__()

        self.conv = ModulatedConv2d(
            in_channels,
            out_channels,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
        )

        # self.noise = NoiseInjection()
        self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        self.activate = FusedLeakyReLU(out_channels)

    def forward(self, input, style, noise=None):
        out = self.conv(input, style)
        # out = self.noise(out, noise=noise)
        out = out + self.bias
        out = self.activate(out)

        return out


class SMPLConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        smpl_dim=85*2,
        demodulate=True,
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        fan_in = in_channels * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.gamma = nn.Linear(smpl_dim, in_channels)
        self.beta = nn.Linear(smpl_dim, in_channels)

        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, kernel_size, kernel_size)
        )

        self.modulation_fc = EqualLinear(smpl_dim, in_channels, bias_init=1)
        # self.modulation_fc1 = EqualLinear(smpl_dim, in_channels, bias_init=1)
        # self.modulation_norm1 = nn.InstanceNorm1d(smpl_dim)
        # self.modulation_norm2 = nn.InstanceNorm1d(in_channels)
        # self.modulation_fc2= EqualLinear(in_channels, in_channels, bias_init=1)
        # self.modulation_relu = nn.ReLU()

        self.demodulate = demodulate

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.in_channels}, {self.out_channels}, {self.kernel_size})"
        )

    def forward(self, input, smpl):
        gamma = self.gamma(smpl).view(smpl.shape[0], -1, 1, 1)
        beta = self.beta(smpl).view(smpl.shape[0], -1, 1, 1)
        input = input * (gamma) + beta

        batch, in_channels, height, width = input.shape
        # smpl = F.relu(smpl)
        # smpl = self.modulation_fc1(smpl)
        # smpl = self.modulation_norm1(smpl)
        # smpl = self.modulation_norm2(smpl)
        # smpl = self.modulation_fc2(smpl)
        # smpl = F.relu(smpl)
        smpl = self.modulation_fc(smpl)
        smpl = smpl.view(batch, 1, in_channels, 1, 1)
        weight = self.scale * self.weight * smpl

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channels, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channels, in_channels, self.kernel_size, self.kernel_size
        )

        input = input.view(1, batch * in_channels, height, width)
        out = conv2d_gradfix.conv2d(
            input, weight, padding=self.padding, groups=batch
        )
        _, _, height, width = out.shape
        out = out.view(batch, self.out_channels, height, width)

        return out


class SMPLConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        smpl_dim=85*2,
        demodulate=True,
    ):
        super().__init__()

        self.conv = SMPLConv2d(
            in_channels,
            out_channels,
            kernel_size,
            smpl_dim,
            demodulate=demodulate,
        )

        # self.noise = NoiseInjection()
        self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        self.activate = FusedLeakyReLU(out_channels)

    def forward(self, input, smpl_params):
        out = self.conv(input, smpl_params)

        out = out + self.bias
        out = self.activate(out)

        return out


class SMPLStyledConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        style_dim,
        smpl_dim=85*2,
        demodulate=True,
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        fan_in = in_channels * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.gamma = nn.Linear(smpl_dim, in_channels)
        self.beta = nn.Linear(smpl_dim, in_channels)

        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, kernel_size, kernel_size)
        )

        self.modulation_fc = EqualLinear(style_dim, in_channels, bias_init=1)
        # self.modulation_norm1 = nn.InstanceNorm1d(smpl_dim)
        # self.modulation_fc1 = EqualLinear(smpl_dim, in_channels, bias_init=1)
        # self.modulation_norm2 = nn.InstanceNorm1d(in_channels)
        # self.modulation_relu = nn.ReLU()
        # self.modulation_fc2= EqualLinear(in_channels, in_channels, bias_init=1)

        self.demodulate = demodulate

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.in_channels}, {self.out_channels}, {self.kernel_size})"
        )

    def forward(self, input, smpl, style):
        gamma = self.gamma(smpl).view(smpl.shape[0], -1, 1, 1)
        beta = self.beta(smpl).view(smpl.shape[0], -1, 1, 1)
        input = input * (gamma) + beta

        batch, in_channels, height, width = input.shape
        # smpl = self.modulation_norm1(smpl)
        # smpl = F.relu(smpl)
        # smpl = self.modulation_fc1(smpl)
        # smpl = self.modulation_norm2(smpl)
        # smpl = F.relu(smpl)
        # smpl = self.modulation_fc2(smpl)
        # smpl = self.modulation_fc(smpl)
        # smpl = smpl.view(batch, 1, in_channels, 1, 1)
        style = self.modulation_fc(style)
        style = style.view(batch, 1, in_channels, 1, 1)
        weight = self.scale * self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channels, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channels, in_channels, self.kernel_size, self.kernel_size
        )

        input = input.view(1, batch * in_channels, height, width)
        out = conv2d_gradfix.conv2d(
            input, weight, padding=self.padding, groups=batch
        )
        _, _, height, width = out.shape
        out = out.view(batch, self.out_channels, height, width)

        return out


class SMPLStyledConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        style_dim,
        smpl_dim,
        demodulate=True,
    ):
        super().__init__()

        self.conv = SMPLStyledConv2d(
            in_channels,
            out_channels,
            kernel_size,
            style_dim,
            smpl_dim,
            demodulate=demodulate,
        )

        # self.noise = NoiseInjection()
        self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        self.activate = FusedLeakyReLU(out_channels)

    def forward(self, input, smpl_params, style):
        out = self.conv(input, smpl_params, style)

        out = out + self.bias
        out = self.activate(out)

        return out

class MyModule(nn.Module):
    def __init__(self, channel):
        super(MyModule, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x, heatmap):
        y = self.conv(heatmap)
        return x * y