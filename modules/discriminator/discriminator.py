import torch
import torch.nn as nn

from modules.discriminator.structure_branch import EdgeDetector, StructureBranch
from modules.discriminator.texture_branch import Discriminator


# class Discriminator(nn.Module):

#     def __init__(self, image_in_channels, edge_in_channels):
#         super(Discriminator, self).__init__()

#         self.texture_branch = TextureBranch(in_channels=image_in_channels)
#         self.structure_branch = StructureBranch(in_channels=edge_in_channels)
#         self.edge_detector = EdgeDetector()

#     def forward(self, output, gray_image, real_edge, is_real=True):

#         if is_real == True:

#             texture_pred = self.texture_branch(output)
#             fake_edge = self.edge_detector(output)
#             structure_pred = self.structure_branch(torch.cat((real_edge, gray_image), dim=1))

#         else:

#             texture_pred = self.texture_branch(output)
#             fake_edge = self.edge_detector(output)
#             structure_pred = self.structure_branch(torch.cat((fake_edge, gray_image), dim=1))

#         return torch.cat((texture_pred, structure_pred), dim=1), fake_edge

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

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm)
        )

        if init_weights:
            self.apply(weights_init())

    def forward(self, image):

        image_pred = self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(image)))))

        if self.use_sigmoid:
            image_pred = torch.sigmoid(image_pred)

        return image_pred