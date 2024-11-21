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

from modules.util import Hourglass, Comb_Hourglass
from modules.util import AntiAliasInterpolation2d, make_coordinate_grid, region2gaussian, kpt2heatmap
from modules.util import to_homogeneous, from_homogeneous
from modules.new_conv import SMPLConv
from utils.pose_utils import smpl2kpts

from SMPLDataset.human_digitalizer.renders import SMPLRenderer
from SMPLDataset.human_digitalizer.bodynets import SMPL


class PixelwiseFlowPredictor(nn.Module):
    def __init__(self, block_expansion, num_blocks, max_features, num_regions, num_channels, scale_factor=1, region_var=0.01,
                 use_covar_heatmap=False, use_deformed_source=True, revert_axis_swap=False,
                 flow=True, srm=True):
        super(PixelwiseFlowPredictor, self).__init__()

        self.num_regions = num_regions
        self.scale_factor = scale_factor
        self.region_var = region_var
        self.use_covar_heatmap = use_covar_heatmap
        self.use_deformed_source = use_deformed_source
        self.revert_axis_swap = revert_axis_swap
        self.kpts = 19
        self.flow = flow
        self.srm = srm


        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(num_channels, self.scale_factor)

        # Unsupservised flow
        depth_channel = 1
        normal_channel = 3
        heatmap_channel = 19


        uns_hourglass_in_features = (num_regions + 1) * num_channels
        self.uns_hourglass = Hourglass(block_expansion=block_expansion, in_features=uns_hourglass_in_features, max_features=max_features, num_blocks=num_blocks, with_mid=True)
        num_unsflow = num_regions + 1
        self.uns_mask = nn.Conv2d(self.uns_hourglass.out_filters, num_unsflow, kernel_size=(7, 7), padding=(3, 3))

        # self.occlusion = nn.Conv2d(self.uns_hourglass.out_filters, 1, kernel_size=(7, 7), padding=(3, 3))

        # Geometry supervised flow
        self.renderer = SMPLRenderer(map_name="par")
        self.smpl_model = SMPL('./SMPLDataset/checkpoints/smpl_model.pkl').eval()
        self.smpl_model.requires_grad_(False)
        self.renderer.set_ambient_light()

        # self.occlusion = nn.Conv2d(3, 1, kernel_size=(7, 7), padding=(3, 3))

        # # End-to-end unsupervised flow
        # e2e_uns_hourglass_in_features = num_channels + (depth_channel + normal_channel + heatmap_channel) * 2
        # self.e2e_uns_hourglass = Hourglass(block_expansion=block_expansion, in_features=e2e_uns_hourglass_in_features, max_features=max_features, num_blocks=num_blocks, with_mid=True)
        # self.flow_regression = nn.Conv2d(self.e2e_uns_hourglass.out_filters, 2, kernel_size=(7, 7), padding=(3, 3))
        # self.occlusion = nn.Conv2d(self.e2e_uns_hourglass.out_filters, 1, kernel_size=(7, 7), padding=(3, 3))


        gs_hourglass_in_features = (self.kpts + 2) * num_channels
        num_gsflow = self.kpts + 2
        self.gs_hourglass = Hourglass(block_expansion=block_expansion, in_features=gs_hourglass_in_features, max_features=max_features, num_blocks=num_blocks, with_mid=True)
        self.gs_mask = nn.Conv2d(self.gs_hourglass.out_filters, num_gsflow, kernel_size=(7, 7), padding=(3, 3))

        # self.occlusion = nn.Conv2d(self.gs_hourglass.out_filters, 1, kernel_size=(7, 7), padding=(3, 3))


        # Combination of unsupervised and geometry supervised flow
        combine_in_features = (num_channels + depth_channel + normal_channel) * 2
        self.combine_net = Comb_Hourglass(block_expansion=block_expansion, in_features=combine_in_features, max_features=max_features, num_blocks=num_blocks)
        self.combine_mask = nn.Conv2d(self.combine_net.out_filters, 1, kernel_size=(7, 7), padding=(3, 3))

        self.occlusion = nn.Conv2d(self.combine_net.out_filters, 1, kernel_size=(7, 7), padding=(3, 3))


    def get_heatmap_representations(self, source_image, driving_region_params, source_region_params):
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

    def get_motion_params_flows(self, source_image, driving_region_params, source_region_params, bg_params=None):
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

    def get_kpt_flows(self, source_image, driving_kpts, source_kpts):
        bs, _, h, w = source_image.shape
        identity_grid = make_coordinate_grid((h, w), type=source_kpts.type())
        identity_grid = identity_grid.view(1, 1, h, w, 2)
        coordinate_grid = identity_grid - driving_kpts.view(bs, self.kpts, 1, 1, 2)
        driving_to_source = coordinate_grid + source_kpts.view(bs, self.kpts, 1, 1, 2)

        #adding background feature
        identity_grid = identity_grid.repeat(bs, 1, 1, 1, 1)
        kpt_flows = torch.cat([identity_grid, driving_to_source], dim=1)
        return kpt_flows

    def get_deformed(self, source_image, sparse_motions):
        bs, _, h, w = source_image.shape
        num_motions = sparse_motions.shape[1]
        source_repeat = source_image.unsqueeze(1).unsqueeze(1).repeat(1, num_motions, 1, 1, 1, 1)
        source_repeat = source_repeat.view(bs * num_motions, -1, h, w)
        sparse_motions = sparse_motions.view(bs * num_motions, h, w, -1)
        sparse_deformed = F.grid_sample(source_repeat, sparse_motions)
        sparse_deformed = sparse_deformed.view(bs, num_motions, -1, h, w)
        return sparse_deformed

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

    def get_smpl_flows(self, source_smpl, driving_smpl):
        cam_from, vert_from = self.get_verts(source_smpl)
        cam_to, vert_to = self.get_verts(driving_smpl)
        f2verts, _, _ = self.renderer.render_fim_wim(cam_from, vert_from)
        f2verts = f2verts[:, :, :, 0:2]

        _, step_fim, step_wim = self.renderer.render_fim_wim(cam_to, vert_to)
        T, occlu_map = self.renderer.cal_bc_transform(f2verts, step_fim, step_wim)

        return T, occlu_map

    def forward(self, source_image, driving_region_params, source_region_params, driving_smpl, source_smpl, bg_params=None, source_smpl_rdr=None, driving_smpl_rdr=None, source_depth=None, driving_depth=None):
        out_dict = dict()
        B,_,H,W = source_image.shape

        driving_smpl = driving_smpl.squeeze(-1)
        source_smpl = source_smpl.squeeze(-1)
        smpl = torch.cat([source_smpl, driving_smpl], dim=1)
        smpl_flow, smpl_mask = self.get_smpl_flows(source_smpl, driving_smpl) # (N,H,W,2), (N,H,W)
        smpl_mask = smpl_mask.unsqueeze(1)
        smpl_warped = F.grid_sample(source_image, smpl_flow) * smpl_mask
        out_dict['smpl_warped'] = smpl_warped

        driving_kpts = smpl2kpts(driving_smpl)
        source_kpts = smpl2kpts(source_smpl)
        kpt_flow = self.get_kpt_flows(source_image, driving_kpts, source_kpts)

        driving_heatmap = kpt2heatmap((driving_kpts+1)/2, spatial_size=(H,W),sigma=3.0)
        source_heatmap = kpt2heatmap((source_kpts+1)/2, spatial_size=(H,W),sigma=3.0)
        out_dict['heatmap'] = driving_heatmap

        if self.scale_factor != 1:
            source_ori = source_image.clone()
            source_image = self.down(source_image)

            driving_heatmap = F.interpolate(driving_heatmap, scale_factor=self.scale_factor, mode='bilinear')
            source_heatmap = F.interpolate(source_heatmap, scale_factor=self.scale_factor, mode='bilinear')

            driving_depth = F.interpolate(driving_depth, scale_factor=self.scale_factor, mode='bilinear')
            source_depth = F.interpolate(source_depth, scale_factor=self.scale_factor, mode='bilinear')

            source_smpl_rdr = F.interpolate(source_smpl_rdr, scale_factor=self.scale_factor, mode='bilinear')
            driving_smpl_rdr = F.interpolate(driving_smpl_rdr, scale_factor=self.scale_factor, mode='bilinear')

            # delta_depth = driving_depth - source_depth
            # delta_heatmap = driving_heatmap - source_heatmap
            # delta_normal = driving_smpl_rdr - source_smpl_rdr

        bs, _, h, w = source_image.shape

        # # end-to-end unsupervised flow only
        # e2e_uns_input = torch.cat([source_image, source_smpl_rdr, driving_smpl_rdr, source_depth, driving_depth, driving_heatmap, source_heatmap], dim=1)
        # e2e_uns_prediction = self.e2e_uns_hourglass(e2e_uns_input)
        # flow = torch.sigmoid(self.flow_regression(e2e_uns_prediction)) * 2 - 1
        # flow_grid = make_coordinate_grid((h, w), type=flow.type())
        # flow = flow_grid + flow.permute(0, 2, 3, 1)
        # out_dict['optical_flow'] = flow
        # occlusion_map = torch.sigmoid(self.occlusion(e2e_uns_prediction))
        # occlusion_map = F.interpolate(occlusion_map, size=(H, W), mode='bilinear')
        # out_dict['occlusion_map'] = occlusion_map

        # # smplflow only
        # out_dict['optical_flow'] = smpl_flow
        # occlusion_map = torch.sigmoid(self.occlusion(smpl_warped))
        # out_dict['occlusion_map'] = F.interpolate(occlusion_map, size=(H, W), mode='bilinear')

        # unsupervised flow
        uns_flow_coarse = self.get_motion_params_flows(source_image, driving_region_params, source_region_params, bg_params=bg_params)
        uns_warped = self.get_deformed(source_image, uns_flow_coarse)
        uns_flow_coarse = uns_flow_coarse.permute(0, 1, 4, 2, 3)
        uns_input = uns_warped.view(bs, -1, h, w)

        uns_prediction = self.uns_hourglass(uns_input)
        uns_mid = self.uns_hourglass.mid

        uns_mask = self.uns_mask(uns_prediction)
        uns_mask = F.softmax(uns_mask, dim=1)
        uns_mask = uns_mask.unsqueeze(2)
        uns_flow_coarse = (uns_flow_coarse * uns_mask).sum(dim=1)
        uns_flow_fine = F.interpolate(uns_flow_coarse, size=(H,W), mode='bilinear')
        uns_warped = F.grid_sample(source_ori, uns_flow_fine.permute(0, 2, 3, 1))
        out_dict['coarse_deformed'] = uns_warped

        # # for unsflow only
        # uns_flow_fine = uns_flow_fine.permute(0, 2, 3, 1)
        # out_dict['optical_flow'] = uns_flow_fine
        # occlusion_map = torch.sigmoid(self.occlusion(uns_prediction))
        # out_dict['occlusion_map'] = F.interpolate(occlusion_map, size=(H, W), mode='bilinear')

        # geometry supervised flow
        kpt_warped = self.get_deformed(source_ori, kpt_flow)
        kpt_warped = kpt_warped.view(bs, -1, H, W)
        kpt_warped_coarse = F.interpolate(kpt_warped, size=(h,w), mode='bilinear')
        smpl_warped_coarse = F.interpolate(smpl_warped, size=(h,w), mode='bilinear')
        gs_input = torch.cat([smpl_warped_coarse, kpt_warped_coarse], dim=1)
        gs_prediction = self.gs_hourglass(gs_input)
        gs_mid = self.gs_hourglass.mid

        gs_mask = self.gs_mask(gs_prediction)
        gs_mask = F.softmax(gs_mask, dim=1)
        gs_mask = F.interpolate(gs_mask, size=(H,W), mode='bilinear')
        gs_mask = gs_mask.unsqueeze(2)

        kpt_flow = kpt_flow.permute(0, 1, 4, 2, 3)
        smpl_flow = smpl_flow.unsqueeze(1).permute(0, 1, 4, 2, 3)
        gs_flow_fine = torch.cat([kpt_flow, smpl_flow], dim=1)
        gs_flow_fine = (gs_flow_fine * gs_mask).sum(dim=1)
        out_dict['fine_deformed'] = F.grid_sample(source_ori, gs_flow_fine.permute(0, 2, 3, 1))

        # # for gsflow only
        # out_dict['optical_flow'] = gs_flow_fine.permute(0, 2, 3, 1)
        # occlusion_map = torch.sigmoid(self.occlusion(gs_prediction))
        # occlusion_map = F.interpolate(occlusion_map, size=(H, W), mode='bilinear')
        # out_dict['occlusion_map'] = occlusion_map

        # combine unsupervised and geometry supervised flow
        coarse_warped = torch.cat([out_dict['coarse_deformed'], out_dict['fine_deformed']], dim=1)
        coarse_warped = F.interpolate(coarse_warped, size=(h,w), mode='bilinear')
        combined_inputs = torch.cat([coarse_warped, driving_depth, source_depth, driving_smpl_rdr, source_smpl_rdr], dim=1)
        # combined_inputs = torch.cat([coarse_warped, driving_depth, source_depth, driving_smpl_rdr, source_smpl_rdr, driving_heatmap, source_heatmap], dim=1)
        combined_outputs = self.combine_net(combined_inputs, uns_mid, gs_mid, smpl)

        combined_mask = self.combine_mask(combined_outputs)
        combined_mask = torch.sigmoid(combined_mask)
        combined_mask = F.interpolate(combined_mask, size=(H, W), mode='bilinear')
        combined_flow = uns_flow_fine * combined_mask + gs_flow_fine * (1 - combined_mask)
        combined_flow = combined_flow.permute(0, 2, 3, 1)
        out_dict['optical_flow'] = combined_flow
        combined_mask = F.interpolate(combined_mask, size=(H, W), mode='bilinear')
        out_dict['combined_mask'] = combined_mask

        occlusion_map = torch.sigmoid(self.occlusion(combined_outputs))
        # occlusion_map = torch.sigmoid(self.occlusion(combined_outputs, smpl_tensor))
        occlusion_map = F.interpolate(occlusion_map, size=(H, W), mode='bilinear')
        out_dict['occlusion_map'] = occlusion_map

        return out_dict