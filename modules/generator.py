"""
Copyright Snap Inc. 2021. This sample code is made available by Snap Inc. for informational purposes only.
No license, whether implied or otherwise, is granted in or to such code (including any rights to copy, modify,
publish, distribute and/or commercialize such code), unless you have entered into a separate agreement for such rights.
Such code is provided as-is, without warranty of any kind, express or implied, including any warranties of merchantability,
title, fitness for a particular purpose, non-infringement, or that such code is free of defects, errors or viruses.
In no event will Snap Inc. be liable for any damages or losses of any kind arising from the sample code or your use thereof.
"""

import torch
from torch import nn
import torch.nn.functional as F
from modules.util import ResBlock2d, SameBlock2d, UpBlock2d, DownBlock2d
from modules.util import SMPLStyledResBlock2d, SMPLStyledUpBlock2d, SMPLStyledSameBlock2d
from modules.util import StyledResBlock2d, StyledUpBlock2d, StyledSameBlock2d
from modules.util import kpt2heatmap
from utils.pose_utils import smpl2kpts
from modules.new_conv import SMPLStyledConv2d, ModulatedConv2d, MyModule
from modules.pixelwise_flow_predictor import PixelwiseFlowPredictor
from modules.backbones import Encoder, QKVLinear, StructBlock
from modules.super_resolution import RCAN

import math


class Generator(nn.Module):
    """
    Generator that given source image and region parameters try to transform image according to movement trajectories
    induced by region parameters. Generator follows Johnson architecture.
    """

    def __init__(self, num_channels, num_regions, block_expansion, max_features, num_down_blocks,
                 num_bottleneck_blocks, pixelwise_flow_predictor_params=None, skips=True, revert_axis_swap=True,mode='conv_concat'):
        super(Generator, self).__init__()
        self.mode = mode.split('_')[0]
        self.num_channels = num_channels
        self.skips = skips
        if pixelwise_flow_predictor_params is not None:
            self.pixelwise_flow_predictor = PixelwiseFlowPredictor(num_regions=num_regions, num_channels=num_channels,
                                                                   revert_axis_swap=revert_axis_swap,
                                                                   mode=mode,
                                                                   **pixelwise_flow_predictor_params)
        else:
            self.pixelwise_flow_predictor = None

        self.encoder = Encoder(in_channels=num_channels+24, block_expansion=block_expansion, max_features=max_features,
                               num_down_blocks=num_down_blocks, skips=skips)

        self.bottleneck = torch.nn.Sequential()
        in_features = min(max_features, block_expansion * (2 ** num_down_blocks))
        for i in range(num_bottleneck_blocks):
            self.bottleneck.add_module('r' + str(i), ResBlock2d(in_features, kernel_size=(3, 3), padding=(1, 1)))

        up_blocks = []
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

    def dot_product_attention(self, query, key, value):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        p_attn = F.softmax(scores, dim = -1)
        return torch.matmul(p_attn, value)

    def apply_optical(self, input_previous=None, input_skip=None, motion_params=None):
        if motion_params is not None:
            if 'occlusion_map' in motion_params:
                occlusion_map = motion_params['occlusion_map']
            else:
                occlusion_map = None
            deformation = motion_params['optical_flow']
            input_skip = self.deform_input(input_skip, deformation)

            if occlusion_map is not None:
                if input_skip.shape[2] != occlusion_map.shape[2] or input_skip.shape[3] != occlusion_map.shape[3]:
                    occlusion_map = F.interpolate(occlusion_map, size=input_skip.shape[2:], mode='bilinear')
                if input_previous is not None:
                    input_skip = input_skip * occlusion_map + input_previous * (1 - occlusion_map)
                else:
                    input_skip = input_skip * occlusion_map
            out = input_skip
        else:
            out = input_previous if input_previous is not None else input_skip
        return out

    def forward(self, source_image, driving_region_params, source_region_params, driving_smpl, source_smpl, bg=None, source_smpl_rdr=None, driving_smpl_rdr=None):
        output_dict = {}
        if self.pixelwise_flow_predictor is not None:
            motion_params = self.pixelwise_flow_predictor(source_image=source_image,
                                                          driving_region_params=driving_region_params,
                                                          source_region_params=source_region_params,
                                                          driving_smpl=driving_smpl, source_smpl=source_smpl,
                                                          bg=bg)
            normal_map = motion_params['normal_map']
            output_dict['normal_map_from'] = motion_params['normal_map_from']
            output_dict['normal_map_to'] = motion_params['normal_map_to']
            output_dict['normal_map'] = motion_params['normal_map']
            output_dict['depth_map'] = motion_params['depth_map']
            output_dict["deformed"] = self.deform_input(source_image, motion_params['optical_flow'])
            output_dict["optical_flow"] = motion_params['optical_flow']
            output_dict['smpl_warped'] = motion_params['smpl_warped']
            if 'combine_mask' in motion_params:
                output_dict['combine_mask'] = motion_params['combine_mask']
            if 'smpl_mask' in motion_params:
                output_dict['smpl_mask'] = motion_params['smpl_mask']
            if 'occlusion_map' in motion_params:
                output_dict['occlusion_map'] = motion_params['occlusion_map']
        else:
            motion_params = None

        if self.mode == 'smplstyle' or self.mode == 'style':
            style = self.style_encoder[0](source_image)
            for i in range(len(self.style_encoder)-1):
                style = self.style_encoder[i+1](style)
            style = F.adaptive_avg_pool2d(style, (1, 1)).view(style.shape[0], -1)

            source_smpl = source_smpl.squeeze(-1)
            driving_smpl = driving_smpl.squeeze(-1)
            smpl = torch.concat([driving_smpl, source_smpl], dim=-1)

        deformed_image = self.deform_input(source_image, motion_params['optical_flow'])
        if deformed_image.shape[2] != motion_params['occlusion_map'].shape[2] or deformed_image.shape[3] != motion_params['occlusion_map'].shape[3]:
            occlusion_map = F.interpolate(motion_params['occlusion_map'], size=deformed_image.shape[2:], mode='bilinear')

        # heatmap = kpt2heatmap(smpl2kpts(driving_smpl.squeeze(-1)), spatial_size=(deformed_image.shape[2], deformed_image.shape[3]),sigma=3.0)
        heatmap = motion_params['heatmap']
        depth_map = motion_params['depth_map']
        inputs = torch.cat([deformed_image, occlusion_map, heatmap, normal_map, depth_map], dim=1)

        skips = self.encoder(inputs)
        out = skips[-1]

        out = self.bottleneck(out)

        output_dict["bottle_neck_feat"] = out

        for i in range(len(self.encoder.blocks)):
            if self.skips:
                out = out + skips[-(i + 1)]

            if self.mode == 'smplstyle':
                out = self.up_blocks[i*2](out, smpl, style)
                out = self.up_blocks[i*2+1](out, smpl, style)
            elif self.mode == 'style':
                out = self.up_blocks[i*2](out, style)
                out = self.up_blocks[i*2+1](out, style)
            else:
                out = self.up_blocks[i*2](out)
                out = self.up_blocks[i*2+1](out)

        if self.skips:
            out = out + skips[0]

        if self.mode == 'smplstyle':
            out = self.final(out, smpl, style)
        elif self.mode == 'style':
            out = self.final(out, style)
        else:
            out = self.final(out)

        out = torch.sigmoid(out)
        output_dict["gen"] = out

        if self.skips:
            out = occlusion_map * deformed_image + (1 - occlusion_map) * out
        output_dict["prediction"] = out

        return output_dict

    def compute_fea(self, source_image):
        out = self.first(source_image)
        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out)
        return out

    def forward_with_flow(self, source_image, optical_flow, occlusion_map):
        out = self.first(source_image)
        skips = [out]
        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out)
            skips.append(out)

        output_dict = {}
        motion_params = {}
        motion_params["optical_flow"] = optical_flow
        motion_params["occlusion_map"] = occlusion_map
        output_dict["deformed"] = self.deform_input(source_image, motion_params['optical_flow'])

        out = self.apply_optical(input_previous=None, input_skip=out, motion_params=motion_params)

        out = self.bottleneck(out)
        for i in range(len(self.up_blocks)):
            if self.skips:
                out = self.apply_optical(input_skip=skips[-(i + 1)], input_previous=out, motion_params=motion_params)
            out = self.up_blocks[i](out)
        if self.skips:
            out = self.apply_optical(input_skip=skips[0], input_previous=out, motion_params=motion_params)
        out = self.final(out)
        out = torch.sigmoid(out)

        if self.skips:
            out = self.apply_optical(input_skip=source_image, input_previous=out, motion_params=motion_params)

        output_dict["prediction"] = out

        return output_dict
