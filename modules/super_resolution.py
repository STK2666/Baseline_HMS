import torch.nn as nn

class RCAN(nn.Module):
    def __init__(self, scale_factor=1, num_channels=3, out_channels=None, num_feats=64, num_blocks=1, num_groups=1):
        super(RCAN, self).__init__()
        self.scale_factor = scale_factor
        if out_channels is None:
            out_channels = num_channels

        self.head = nn.Conv2d(num_channels, num_feats, kernel_size=3, padding=1)

        self.body = RCABlock(num_feats)

        self.tail = nn.Sequential(
            # nn.Conv2d(num_feats, num_feats * scale_factor ** 2, kernel_size=3, padding=1),
            # nn.PixelShuffle(scale_factor),
            nn.Conv2d(num_feats, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x

class RCABlock(nn.Module):
    def __init__(self, num_feats):
        super(RCABlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(num_feats, num_feats, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feats, num_feats, kernel_size=3, padding=1)
        )
        self.ca = CALayer(num_feats)

    def forward(self, x):
        res = self.block(x)
        res = self.ca(res)
        return x + res

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y