from torch import nn
from modules.util import SameBlock2d, DownBlock2d


class Encoder(nn.Module):
    """
    Generator that given source image and region parameters try to transform image according to movement trajectories
    induced by region parameters. Generator follows Johnson architecture.
    """

    def __init__(self, in_channels, block_expansion, max_features, num_down_blocks,
                skips=True):
        super(Encoder, self).__init__()
        self.in_channels = in_channels
        self.skips = skips
        self.fromRGB = SameBlock2d(in_channels, block_expansion, kernel_size=(7, 7), padding=(3, 3))
        blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        x = self.fromRGB(x)
        skips = [x]
        for block in self.blocks:
            x = block(x)
            skips.append(x)
        return skips if self.skips else x


class QKVLinear(nn.Module):
    def __init__(self, block_expansion, max_features, num_up_blocks):
        super(QKVLinear, self).__init__()
        blocks = []
        for i in range(num_up_blocks):
            features = min(max_features, block_expansion * (2 ** (num_up_blocks - i)))
            blocks.append(SameBlock2d(features, features, kernel_size=(1, 1), padding=(0, 0)))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x_list):
        out_list = []
        for i in range(len(self.blocks)):
            out_list.append(self.blocks[i](x_list[i]))
        return out_list