import torch
import torch.nn as nn
from modules.misc import weights_init, spectral_norm


class Discriminator(nn.Module):

    def __init__(self, in_channels=3, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__()

        self.use_sigmoid = use_sigmoid

        self.conv1 = self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # self.conv5 = nn.Sequential(
        #     spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1,
        #                             bias=not use_spectral_norm), use_spectral_norm)
        # )

        if init_weights:
            self.apply(weights_init())

    def forward(self, image):

        image_pred = self.conv4(self.conv3(self.conv2(self.conv1(image))))
        # image_pred = self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(image)))))

        if self.use_sigmoid:
            image_pred = torch.sigmoid(image_pred)

        return image_pred


class ResDiscriminator(nn.Module):
    """
    ResNet Discriminator Network
    :param input_nc: number of channels in input
    :param ndf: base filter channel
    :param img_f: the largest feature channels
    :param layers: down and up sample layers
    :param norm: normalization function 'instance, batch, group'
    :param activation: activation function 'ReLU, SELU, LeakyReLU, PReLU'
    """
    def __init__(self, input_nc=3, ndf=32, img_f=128, layers=4, norm='none', activation='LeakyReLU', use_spect=True,
                 use_coord=False, init_weights=True):
        super(ResDiscriminator, self).__init__()

        self.layers = layers

        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)
        self.nonlinearity = nonlinearity

        # encoder part
        self.block0 = ResBlockEncoder(input_nc, ndf, ndf, norm_layer, nonlinearity, use_spect, use_coord)

        mult = 1
        for i in range(layers - 1):
            mult_prev = mult
            mult = min(2 ** (i + 1), img_f // ndf)
            block = ResBlockEncoder(ndf * mult_prev, ndf * mult, ndf * mult_prev, norm_layer, nonlinearity, use_spect, use_coord)
            setattr(self, 'encoder' + str(i), block)
        self.conv = spectral_norm(nn.Conv2d(ndf * mult, 1, 1), use_spect)

        if init_weights:
            self.apply(weights_init())

    def forward(self, x):
        out = self.block0(x)
        for i in range(self.layers - 1):
            model = getattr(self, 'encoder' + str(i))
            out = model(out)
        out = self.conv(self.nonlinearity(out))
        out = torch.sigmoid(out)
        return out

class ResBlockEncoder(nn.Module):
    """
    Define a decoder block
    """
    def __init__(self, input_nc, output_nc, hidden_nc=None, norm_layer=nn.BatchNorm2d, nonlinearity=nn.LeakyReLU(),
                 use_spect=False, use_coord=False):
        super(ResBlockEncoder, self).__init__()

        hidden_nc = input_nc if hidden_nc is None else hidden_nc

        conv1 = spectral_norm(nn.Conv2d(input_nc, hidden_nc, kernel_size=3, stride=1, padding=1), use_spect)
        conv2 = spectral_norm(nn.Conv2d(hidden_nc, output_nc, kernel_size=4, stride=2, padding=1), use_spect)
        bypass = spectral_norm(nn.Conv2d(input_nc, output_nc, kernel_size=1, stride=1, padding=0), use_spect)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(nonlinearity, conv1, nonlinearity, conv2,)
        else:
            self.model = nn.Sequential(norm_layer(input_nc), nonlinearity, conv1,
                                       norm_layer(hidden_nc), nonlinearity, conv2,)
        self.shortcut = nn.Sequential(nn.AvgPool2d(kernel_size=2, stride=2), bypass)

    def forward(self, x):
        out = self.model(x) + self.shortcut(x)
        return out

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = nn.BatchNorm2d
    elif norm_type == 'instance':
        norm_layer = nn.InstanceNorm2d
    elif norm_type == 'group':
        norm_layer = lambda num_features: nn.GroupNorm(32, num_features)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def get_nonlinearity_layer(activation_type='LeakyReLU'):
    if activation_type == 'ReLU':
        nonlinearity_layer = nn.ReLU()
    elif activation_type == 'LeakyReLU':
        nonlinearity_layer = nn.LeakyReLU(0.2, True)
    elif activation_type == 'PReLU':
        nonlinearity_layer = nn.PReLU()
    elif activation_type == 'SELU':
        nonlinearity_layer = nn.SELU()
    else:
        raise NotImplementedError('activation layer [%s] is not found' % activation_type)
    return nonlinearity_layer