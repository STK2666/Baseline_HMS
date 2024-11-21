"""
Copyright Snap Inc. 2021. This sample code is made available by Snap Inc. for informational purposes only.
No license, whether implied or otherwise, is granted in or to such code (including any rights to copy, modify,
publish, distribute and/or commercialize such code), unless you have entered into a separate agreement for such rights.
Such code is provided as-is, without warranty of any kind, express or implied, including any warranties of merchantability,
title, fitness for a particular purpose, non-infringement, or that such code is free of defects, errors or viruses.
In no event will Snap Inc. be liable for any damages or losses of any kind arising from the sample code or your use thereof.

Modified by: Tongkai Shi
"""



import torch
from torch import nn
import torch.nn.functional as F
from modules.util import ResBlock2d, SameBlock2d, UpBlock2d, StyledUpBlock2d, StyledSameBlock2d
from modules.pixelwise_flow_predictor import PixelwiseFlowPredictor
from modules.backbones import Encoder
from modules.pretrained import PartialResNet50


def dot_product_attention(q, k, v):
    embed_dim = q.shape[1]
    scale = embed_dim ** -0.5
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn = F.softmax(attn, dim=-1)
    return torch.matmul(attn, v)


class Generator(nn.Module):
    """
    Generator that given source image and region parameters try to transform image according to movement trajectories
    induced by region parameters. Generator follows Johnson architecture.
    """

    def __init__(self, num_channels, num_regions, block_expansion, max_features, num_down_blocks,
                 num_bottleneck_blocks, pixelwise_flow_predictor_params=None, skips=True, revert_axis_swap=True,
                 flow=True, blend=False, mode='concat'):
        super(Generator, self).__init__()
        # self.mode = mode.split('_')[0]
        self.num_channels = num_channels
        self.skips = skips
        self.flow = flow
        self.blend = blend
        self.mode = mode
        if pixelwise_flow_predictor_params is not None:
            self.pixelwise_flow_predictor = PixelwiseFlowPredictor(num_regions=num_regions, num_channels=num_channels,
                                                                   revert_axis_swap=revert_axis_swap,
                                                                   flow=flow,
                                                                   **pixelwise_flow_predictor_params)
        else:
            self.pixelwise_flow_predictor = None

        # encoder
        if self.flow:
            # self.encoder = Encoder(in_channels=num_channels+33, block_expansion=block_expansion, max_features=max_features,
            # self.encoder = Encoder(in_channels=num_channels+24, block_expansion=block_expansion, max_features=max_features,
            #                      num_down_blocks=num_down_blocks, skips=skips)
            self.encoder = nn.Sequential(
                nn.Conv2d(num_channels+24, 3, kernel_size=(7, 7), padding=(3, 3)),
                PartialResNet50(),
                # nn.Conv2d(512, block_expansion, kernel_size=(3, 3), padding=(1, 1)),
                nn.ReLU(),
                nn.Conv2d(512, max_features, kernel_size=(3, 3), padding=(1, 1)),
            )
            self.frozen_resnet()
            
        elif self.mode == 'concat':
            self.encoder = Encoder(in_channels=26, block_expansion=block_expansion, max_features=max_features,
                                 num_down_blocks=num_down_blocks, skips=skips)
        else:
            self.encoder = Encoder(in_channels=23, block_expansion=block_expansion, max_features=max_features,
                                 num_down_blocks=num_down_blocks, skips=skips)

        if self.mode == 'att':
            self.appearance_encoder = Encoder(in_channels=num_channels, block_expansion=block_expansion, max_features=max_features,
                                 num_down_blocks=num_down_blocks, skips=skips)
        elif self.mode == 'style':
            self.style_encoder = Encoder(in_channels=num_channels, block_expansion=block_expansion, max_features=max_features*2,
                                     num_down_blocks=num_down_blocks+2, skips=False)

        self.bottleneck = torch.nn.Sequential()
        in_features = min(max_features, block_expansion * (2 ** num_down_blocks))
        for i in range(num_bottleneck_blocks):
            self.bottleneck.add_module('r' + str(i), ResBlock2d(in_features, kernel_size=(3, 3), padding=(1, 1)))

        up_blocks = []
        if self.mode == 'att':
            q_blocks = []
            k_blocks = []
            v_blocks = []
            for i in range(num_down_blocks):
                in_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i)))
                out_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i - 1)))
                q_blocks.append(SameBlock2d(in_features, in_features, kernel_size=(1, 1), padding=(0, 0)))
                k_blocks.append(SameBlock2d(in_features, in_features, kernel_size=(1, 1), padding=(0, 0)))
                v_blocks.append(SameBlock2d(in_features, in_features, kernel_size=(1, 1), padding=(0, 0)))
                up_blocks.append(UpBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
                up_blocks.append(SameBlock2d(out_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
            self.q_blocks = nn.ModuleList(q_blocks)
            self.k_blocks = nn.ModuleList(k_blocks)
            self.v_blocks = nn.ModuleList(v_blocks)

        elif self.mode == 'style':
            for i in range(num_down_blocks):
                in_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i)))
                out_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i - 1)))
                up_blocks.append(StyledUpBlock2d(in_features, out_features, style_dim=max_features*2, kernel_size=3))
                up_blocks.append(StyledSameBlock2d(out_features, out_features, style_dim=max_features*2, kernel_size=3))
        else:
            for i in range(num_down_blocks):
                in_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i)))
                out_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i - 1)))
                up_blocks.append(UpBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
                up_blocks.append(SameBlock2d(out_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.up_blocks = nn.ModuleList(up_blocks)

        self.final = nn.Conv2d(block_expansion, num_channels, kernel_size=(7, 7), padding=(3, 3))


    @staticmethod
    def deform_input(inp, optical_flow):
        _, h_old, w_old, _ = optical_flow.shape
        _, _, h, w = inp.shape
        if h_old != h or w_old != w:
            optical_flow = optical_flow.permute(0, 3, 1, 2)
            optical_flow = F.interpolate(optical_flow, size=(h, w), mode='bilinear')
            optical_flow = optical_flow.permute(0, 2, 3, 1)
        return F.grid_sample(inp, optical_flow)

    def frozen_resnet(self):
        for param in self.encoder[1].parameters():
            param.requires_grad = False

    def forward(self, source_image, driving_region_params, source_region_params, driving_smpl, source_smpl, bg_params=None, source_smpl_rdr=None, driving_smpl_rdr=None, source_depth=None, driving_depth=None):
        output_dict = {}

        # flow prediction
        motion_params = self.pixelwise_flow_predictor(source_image=source_image, driving_region_params=driving_region_params, source_region_params=source_region_params, bg_params=bg_params,
                                                      driving_smpl=driving_smpl, source_smpl=source_smpl, source_smpl_rdr=source_smpl_rdr, driving_smpl_rdr=driving_smpl_rdr,
                                                      source_depth=source_depth, driving_depth=driving_depth)
        for key in motion_params:
            output_dict[key] = motion_params[key]

        # generator inputs
        heatmap = motion_params['heatmap']
        inputs = torch.cat([heatmap, driving_smpl_rdr, driving_depth], dim=1)
        if self.flow:
            deformed_image = self.deform_input(source_image, motion_params['optical_flow'])
            output_dict["deformed"] = deformed_image
            output_dict['driving_smpl_rdr'] = driving_smpl_rdr
            occlusion_map = motion_params['occlusion_map']
            if deformed_image.shape[2] != motion_params['occlusion_map'].shape[2] or deformed_image.shape[3] != motion_params['occlusion_map'].shape[3]:
                occlusion_map = F.interpolate(occlusion_map, size=deformed_image.shape[2:], mode='bilinear')
            inputs = torch.cat([deformed_image, occlusion_map, inputs], dim=1)
        elif self.mode == 'concat':
            inputs = torch.cat([source_image, inputs], dim=1)

        # encode
        skips = self.encoder(inputs)
        # out = skips[-1]
        out = skips
        if self.mode == 'att':
            app = self.appearance_encoder(source_image)
        elif self.mode == 'style':
            style = self.style_encoder(source_image)
            style = F.adaptive_avg_pool2d(style, (1, 1)).view(style.shape[0], -1)

        # bottleneck
        out = self.bottleneck(out)
        output_dict["bottle_neck_feat"] = out

        # decode
        for i in range(int(len(self.up_blocks)/2)):
            # if self.skips:
                # out = out + skips[-(i + 1)]

            if self.mode == 'att':
                q = self.q_blocks[i](out)
                k = self.k_blocks[i](app[-(i + 1)])
                v = self.v_blocks[i](app[-(i + 1)])
                out = out + dot_product_attention(q, k, v)

            if self.mode == 'style':
                out = self.up_blocks[i*2](out, style)
                out = self.up_blocks[i*2+1](out, style)
            else:
                out = self.up_blocks[i*2](out)
                out = self.up_blocks[i*2+1](out)

        # if self.skips:
            # out = out + skips[0]

        # to rgb
        out = torch.sigmoid(self.final(out))
        output_dict["gen"] = out

        # blend with deformed
        if self.blend:
            out = occlusion_map * deformed_image + (1 - occlusion_map) * out
        output_dict["prediction"] = out

        return output_dict