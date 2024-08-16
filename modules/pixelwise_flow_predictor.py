"""
Copyright Snap Inc. 2021. This sample code is made available by Snap Inc. for informational purposes only.
No license, whether implied or otherwise, is granted in or to such code (including any rights to copy, modify,
publish, distribute and/or commercialize such code), unless you have entered into a separate agreement for such rights.
Such code is provided as-is, without warranty of any kind, express or implied, including any warranties of merchantability,
title, fitness for a particular purpose, non-infringement, or that such code is free of defects, errors or viruses.
In no event will Snap Inc. be liable for any damages or losses of any kind arising from the sample code or your use thereof.
"""

from torch import nn
import torch.nn.functional as F
import torch
import numpy as np

from modules.util import Hourglass, AntiAliasInterpolation2d, make_coordinate_grid, region2gaussian
from modules.util import to_homogeneous, from_homogeneous
from utils.pose_utils import kpt2heatmap, smpl2kpts

from modules.new_conv import SMPLConv
from SMPLDataset.human_digitalizer.renders import SMPLRenderer
from SMPLDataset.human_digitalizer.bodynets import SMPL


class PixelwiseFlowPredictor(nn.Module):
    """
    Module that predicts a pixelwise flow from sparse motion representation given by
    source_region_params and driving_region_params
    """

    def __init__(self, block_expansion, num_blocks, max_features, num_regions, num_channels,
                 estimate_occlusion_map=False, scale_factor=1, region_var=0.01,
                 use_covar_heatmap=False, use_deformed_source=True, revert_axis_swap=False,
                 mode='conv_concat', unsupervised_flow=False, smpl_rdr_input=False):
        super(PixelwiseFlowPredictor, self).__init__()
        self.conv_mode = mode.split('_')[0]
        self.flow_mode = mode.split('_')[1]

        self.num_regions = num_regions
        self.scale_factor = scale_factor
        self.region_var = region_var
        self.use_covar_heatmap = use_covar_heatmap
        self.use_deformed_source = use_deformed_source
        self.revert_axis_swap = revert_axis_swap
        self.unsupervised_flow = unsupervised_flow
        self.smpl_rdr_input = smpl_rdr_input


        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(num_channels, self.scale_factor)

        # if self.smpl_rdr_input:
            # hourglass_in_features = (num_regions + 1) * (num_channels * use_deformed_source + 1) + 2*num_channels
        depth_channel = 1
        normal_channel = 3
        heatmap_channel = 19
        hourglass_in_features = (num_regions + 1) * (num_channels * use_deformed_source + 1) + num_channels + depth_channel + normal_channel + heatmap_channel
        self.hourglass = Hourglass(block_expansion=block_expansion,
                                   in_features=hourglass_in_features,
                                   max_features=max_features, num_blocks=num_blocks)

        # if self.unsupervised_flow:
        #     uns_hourglass = Hourglass(block_expansion=block_expansion, in_features=num_channels*3,
        #                                 max_features=max_features, num_blocks=num_blocks)
        #     self.uns_flow = nn.Sequential(uns_hourglass,
        #                                   nn.Conv2d(uns_hourglass.out_filters, 2, kernel_size=(7, 7), padding=(3, 3)))


        self.renderer = SMPLRenderer(map_name="par")
        self.smpl_model = SMPL('./SMPLDataset/checkpoints/smpl_model.pkl').eval()
        self.smpl_model.requires_grad_(False)
        self.renderer.set_ambient_light()

        num_flows = num_regions + 1 if self.flow_mode != 'concat' else num_regions + 2
        num_flows = num_flows + 1 if self.unsupervised_flow else num_flows
        self.mask = nn.Conv2d(self.hourglass.out_filters, num_flows, kernel_size=(7, 7), padding=(3, 3))

        if estimate_occlusion_map:
            self.occlusion = nn.Conv2d(self.hourglass.out_filters, 1, kernel_size=(7, 7), padding=(3, 3))
        else:
            self.occlusion = None


    def create_heatmap_representations(self, source_image, driving_region_params, source_region_params):
        """
        Eq 6. in the paper H_k(z)
        """
        spatial_size = source_image.shape[2:]
        covar = self.region_var if not self.use_covar_heatmap else driving_region_params['covar']
        gaussian_driving = region2gaussian(driving_region_params['shift'], covar=covar, spatial_size=spatial_size)
        covar = self.region_var if not self.use_covar_heatmap else source_region_params['covar']
        gaussian_source = region2gaussian(source_region_params['shift'], covar=covar, spatial_size=spatial_size)

        heatmap = gaussian_driving - gaussian_source

        # adding background feature
        zeros = torch.zeros(heatmap.shape[0], 1, spatial_size[0], spatial_size[1])
        heatmap = torch.cat([zeros.type(heatmap.type()), heatmap], dim=1)
        heatmap = heatmap.unsqueeze(2)
        return heatmap

    def create_sparse_motions(self, source_image, driving_region_params, source_region_params, bg_params=None):
        bs, _, h, w = source_image.shape
        identity_grid = make_coordinate_grid((h, w), type=source_region_params['shift'].type())
        identity_grid = identity_grid.view(1, 1, h, w, 2)
        coordinate_grid = identity_grid - driving_region_params['shift'].view(bs, self.num_regions, 1, 1, 2)
        if 'affine' in driving_region_params:
            # affine = torch.matmul(source_region_params['affine'], torch.inverse(driving_region_params['affine']))
            try:
                affine = torch.matmul(source_region_params['affine'], torch.inverse(driving_region_params['affine']))
            except torch._C._LinAlgError:
                replaced = driving_region_params['affine']
                # replaced = torch.where(replaced==0,replaced.min(),replaced)
                replaced = torch.linalg.pinv(replaced)
                affine = torch.matmul(source_region_params['affine'], replaced)
            if self.revert_axis_swap:
                affine = affine * torch.sign(affine[:, :, 0:1, 0:1])
            affine = affine.unsqueeze(-3).unsqueeze(-3)
            affine = affine.repeat(1, 1, h, w, 1, 1)
            coordinate_grid = torch.matmul(affine, coordinate_grid.unsqueeze(-1))
            coordinate_grid = coordinate_grid.squeeze(-1)

        driving_to_source = coordinate_grid + source_region_params['shift'].view(bs, self.num_regions, 1, 1, 2)

        # adding background feature
        if bg_params is None:
            bg_grid = identity_grid.repeat(bs, 1, 1, 1, 1)
        else:
            bg_grid = identity_grid.repeat(bs, 1, 1, 1, 1)
            bg_grid = to_homogeneous(bg_grid)
            bg_grid = torch.matmul(bg_params.view(bs, 1, 1, 1, 3, 3), bg_grid.unsqueeze(-1)).squeeze(-1)
            bg_grid = from_homogeneous(bg_grid)

        sparse_motions = torch.cat([bg_grid, driving_to_source], dim=1)

        return sparse_motions

    def create_deformed_source_image(self, source_image, sparse_motions):
        bs, _, h, w = source_image.shape
        source_repeat = source_image.unsqueeze(1).unsqueeze(1).repeat(1, self.num_regions + 1, 1, 1, 1, 1)
        source_repeat = source_repeat.view(bs * (self.num_regions + 1), -1, h, w)
        sparse_motions = sparse_motions.view((bs * (self.num_regions + 1), h, w, -1))
        sparse_deformed = F.grid_sample(source_repeat, sparse_motions)
        sparse_deformed = sparse_deformed.view((bs, self.num_regions + 1, -1, h, w))
        return sparse_deformed


    def get_normals(self, cams, verts):
        # get normal map (-1, 1)
        normal_map = self.renderer.render_normal_map(cams, verts)
        normal_map = normal_map*2 -1
        return normal_map

    def get_verts(self, smpl_para, get_landmarks=False):
        cam_nc = 3
        pose_nc = 72
        shape_nc = 10

        cam = smpl_para[:, 0:cam_nc].contiguous()
        pose = smpl_para[:, cam_nc:cam_nc + pose_nc].contiguous()
        shape = smpl_para[:, -shape_nc:].contiguous()
        # with torch.no_grad():
        verts, kpts3d, _ = self.smpl_model(beta=shape, theta=pose, get_skin=True)
        if get_landmarks:
            X_trans = kpts3d[:, :, :2] + cam[:, None, 1:]
            kpts2d = cam[:, None, 0:1] * X_trans
            return cam, verts, kpts2d
        else:
            return cam, verts

    def get_flow(self, source_smpl, driving_smpl):
        cam_from, vert_from = self.get_verts(source_smpl)
        cam_to, vert_to = self.get_verts(driving_smpl)

        f2verts, _, _ = self.renderer.render_fim_wim(cam_from, vert_from)
        f2verts = f2verts[:, :, :, 0:2]

        _, step_fim, step_wim = self.renderer.render_fim_wim(cam_to, vert_to)
        T, occlu_map = self.renderer.cal_bc_transform(f2verts, step_fim, step_wim)

        return T, occlu_map

    def forward(self, source_image, driving_region_params, source_region_params, driving_smpl, source_smpl, bg_params=None, source_smpl_rdr=None, driving_smpl_rdr=None, source_depth=None, driving_depth=None):
        out_dict = dict()
        driving_smpl = driving_smpl.squeeze(-1)
        source_smpl = source_smpl.squeeze(-1)

        smpl_flow, smpl_mask = self.get_flow(source_smpl, driving_smpl) # (N,H,W,2), (N,H,W)

        smpl_mask = smpl_mask.unsqueeze(1)
        smpl_warped = F.grid_sample(source_image, smpl_flow) * smpl_mask
        out_dict['smpl_warped'] = smpl_warped

        driving_heatmap = kpt2heatmap(smpl2kpts(driving_smpl), spatial_size=(source_image.shape[2], source_image.shape[3]),sigma=3.0)
        source_heatmap = kpt2heatmap(smpl2kpts(source_smpl), spatial_size=(source_image.shape[2], source_image.shape[3]),sigma=3.0)
        out_dict['heatmap'] = driving_heatmap

        if self.scale_factor != 1:
            source_image = self.down(source_image)
            smpl_warped = self.down(smpl_warped)

            driving_heatmap = F.interpolate(driving_heatmap, size=(smpl_warped.shape[2], smpl_warped.shape[3]), mode='bilinear')
            source_heatmap = F.interpolate(source_heatmap, size=(smpl_warped.shape[2], smpl_warped.shape[3]), mode='bilinear')

            drving_depth = F.interpolate(drving_depth, size=(smpl_warped.shape[2], smpl_warped.shape[3]), mode='bilinear')
            source_depth = F.interpolate(source_depth, size=(smpl_warped.shape[2], smpl_warped.shape[3]), mode='bilinear')

            source_smpl_rdr = F.interpolate(source_smpl_rdr, size=(smpl_warped.shape[2], smpl_warped.shape[3]), mode='bilinear')
            driving_smpl_rdr = F.interpolate(driving_smpl_rdr, size=(smpl_warped.shape[2], smpl_warped.shape[3]), mode='bilinear')

            smpl_flow = F.interpolate(smpl_flow.permute(0, 3, 1, 2), size=(smpl_warped.shape[2], smpl_warped.shape[3]), mode='bilinear').permute(0, 2, 3, 1)

        bs, _, h, w = source_image.shape
        heatmap_representation = self.create_heatmap_representations(source_image, driving_region_params, source_region_params)

        delta_depth = drving_depth - source_depth
        delta_heatmap = driving_heatmap - source_heatmap
        delta_normal = driving_smpl_rdr - source_smpl_rdr

        sparse_motion = self.create_sparse_motions(source_image, driving_region_params, source_region_params, bg_params=bg_params)
        deformed_source = self.create_deformed_source_image(source_image, sparse_motion)

        sparse_motion = sparse_motion.permute(0, 1, 4, 2, 3)
        smpl_flow = smpl_flow.unsqueeze(-1).permute(0, 4, 3, 1, 2)
        sparse_motion = torch.cat([sparse_motion, smpl_flow], dim=1)

        if self.use_deformed_source:
            predictor_input = torch.cat([heatmap_representation, deformed_source], dim=2) # region_heatmaps and deformed_source_features
        else:
            predictor_input = heatmap_representation

        predictor_input = predictor_input.view(bs, -1, h, w)
        predictor_input = torch.cat([predictor_input, smpl_warped, delta_heatmap, delta_depth, delta_normal], dim=1)
        prediction = self.hourglass(predictor_input)

        mask = self.mask(prediction)
        mask = F.softmax(mask, dim=1)
        mask = mask.unsqueeze(2)
        deformation = (sparse_motion * mask).sum(dim=1)
        deformation = deformation.permute(0, 2, 3, 1)
        out_dict['optical_flow'] = deformation

        if self.occlusion:
            occlusion_map = torch.sigmoid(self.occlusion(prediction))
            out_dict['occlusion_map'] = occlusion_map

        return out_dict
