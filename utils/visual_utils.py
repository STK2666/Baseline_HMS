import os
from turtle import back
from mim import run
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from logger import Logger
import numpy as np
import imageio
import math
import cv2
from skimage.draw import circle
from matplotlib import image, pyplot as plt
from skimage import io, img_as_float32
from torch.nn.functional import interpolate


def flow2img(flow_data):
    """
    convert optical flow into color image
    :param flow_data:
    :return: color image
    """
    # print(flow_data.shape)
    # print(type(flow_data))
    u = flow_data[:, :, 0]
    v = flow_data[:, :, 1]

    UNKNOW_FLOW_THRESHOLD = 1e1
    pr1 = abs(u) > UNKNOW_FLOW_THRESHOLD
    pr2 = abs(v) > UNKNOW_FLOW_THRESHOLD
    idx_unknown = (pr1 | pr2)
    u[idx_unknown] = v[idx_unknown] = 0

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))
    u = u / maxrad + np.finfo(float).eps
    v = v / maxrad + np.finfo(float).eps

    img = compute_color(u, v)
    idx = np.repeat(idx_unknown[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    idx_over_flow = img > 255
    img[idx_over_flow] = 255
    idx_under_flow = img < 0
    img[idx_under_flow] = 0

    return np.uint8(img)


def compute_color(u, v):
	"""
	compute optical flow color map
	:param u: horizontal optical flow
	:param v: vertical optical flow
	:return:
	"""

	height, width = u.shape
	img = np.zeros((height, width, 3))

	NAN_idx = np.isnan(u) | np.isnan(v)
	u[NAN_idx] = v[NAN_idx] = 0

	colorwheel = make_color_wheel()
	ncols = np.size(colorwheel, 0)

	rad = np.sqrt(u ** 2 + v ** 2)

	a = np.arctan2(-v, -u) / np.pi

	fk = (a + 1) / 2 * (ncols - 1) + 1

	k0 = np.floor(fk).astype(int)

	k1 = k0 + 1
	k1[k1 == ncols + 1] = 1
	f = fk - k0

	for i in range(0, np.size(colorwheel, 1)):
		tmp = colorwheel[:, i]
		col0 = tmp[k0 - 1] / 255
		col1 = tmp[k1 - 1] / 255
		col = (1 - f) * col0 + f * col1

		idx = rad <= 1
		col[idx] = 1 - rad[idx] * (1 - col[idx])
		notidx = np.logical_not(idx)

		col[notidx] *= 0.75
		img[:, :, i] = np.uint8(np.floor(255 * col * (1 - NAN_idx)))

	return img


def make_color_wheel():
	"""
	Generate color wheel according Middlebury color code
	:return: Color wheel
	"""
	RY = 15
	YG = 6
	GC = 4
	CB = 11
	BM = 13
	MR = 6

	ncols = RY + YG + GC + CB + BM + MR

	colorwheel = np.zeros([ncols, 3])

	col = 0

	# RY
	colorwheel[0:RY, 0] = 255
	colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
	col += RY

	# YG
	colorwheel[col:col + YG, 0] = 255 - np.transpose(np.floor(255 * np.arange(0, YG) / YG))
	colorwheel[col:col + YG, 1] = 255
	col += YG

	# GC
	colorwheel[col:col + GC, 1] = 255
	colorwheel[col:col + GC, 2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
	col += GC

	# CB
	colorwheel[col:col + CB, 1] = 255 - np.transpose(np.floor(255 * np.arange(0, CB) / CB))
	colorwheel[col:col + CB, 2] = 255
	col += CB

	# BM
	colorwheel[col:col + BM, 2] = 255
	colorwheel[col:col + BM, 0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
	col += + BM

	# MR
	colorwheel[col:col + MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
	colorwheel[col:col + MR, 0] = 255

	return colorwheel


# def make_color_wheel(bins=None):
#     """Build a color wheel.

#     Args:
#         bins(list or tuple, optional): Specify the number of bins for each
#             color range, corresponding to six ranges: red -> yellow,
#             yellow -> green, green -> cyan, cyan -> blue, blue -> magenta,
#             magenta -> red. [15, 6, 4, 11, 13, 6] is used for default
#             (see Middlebury).

#     Returns:
#         ndarray: Color wheel of shape (total_bins, 3).
#     """
#     if bins is None:
#         bins = [15, 6, 4, 11, 13, 6]
#     assert len(bins) == 6

#     RY, YG, GC, CB, BM, MR = tuple(bins)

#     ry = [1, np.arange(RY) / RY, 0]
#     yg = [1 - np.arange(YG) / YG, 1, 0]
#     gc = [0, 1, np.arange(GC) / GC]
#     cb = [0, 1 - np.arange(CB) / CB, 1]
#     bm = [np.arange(BM) / BM, 0, 1]
#     mr = [1, 0, 1 - np.arange(MR) / MR]

#     num_bins = RY + YG + GC + CB + BM + MR

#     color_wheel = np.zeros((3, num_bins), dtype=np.float32)

#     col = 0
#     for i, color in enumerate([ry, yg, gc, cb, bm, mr]):
#         for j in range(3):
#             color_wheel[j, col:col + bins[i]] = color[j]
#         col += bins[i]

#     return color_wheel.T

def flow2rgb(flow, color_wheel=None, unknown_thr=1e6):
    """Convert flow map to RGB image.

    Args:
        flow (ndarray): Array of optical flow.
        color_wheel (ndarray or None): Color wheel used to map flow field to
            RGB colorspace. Default color wheel will be used if not specified.
        unknown_thr (str): Values above this threshold will be marked as
            unknown and thus ignored.

    Returns:
        ndarray: RGB image that can be visualized.
    """
    assert flow.ndim == 3 and flow.shape[-1] == 2
    if color_wheel is None:
        color_wheel = make_color_wheel()
    assert color_wheel.ndim == 2 and color_wheel.shape[1] == 3
    num_bins = color_wheel.shape[0]

    dx = flow[:, :, 0].copy()
    dy = flow[:, :, 1].copy()

    ignore_inds = (
        np.isnan(dx) | np.isnan(dy) | (np.abs(dx) > unknown_thr) |
        (np.abs(dy) > unknown_thr))
    dx[ignore_inds] = 0
    dy[ignore_inds] = 0

    rad = np.sqrt(dx**2 + dy**2)
    if np.any(rad > np.finfo(float).eps):
        max_rad = np.max(rad)
        dx /= max_rad
        dy /= max_rad

    rad = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(-dy, -dx) / np.pi

    bin_real = (angle + 1) / 2 * (num_bins - 1)
    bin_left = np.floor(bin_real).astype(int)
    bin_right = (bin_left + 1) % num_bins
    w = (bin_real - bin_left.astype(np.float32))[..., None]
    flow_img = (1 -
                w) * color_wheel[bin_left, :] + w * color_wheel[bin_right, :]
    small_ind = rad <= 1
    flow_img[small_ind] = 1 - rad[small_ind, None] * (1 - flow_img[small_ind])
    flow_img[np.logical_not(small_ind)] *= 0.75

    flow_img[ignore_inds, :] = 0

    return flow_img

def draw_image_with_kp(image, kp_array):
    image = np.copy(image)
    spatial_size = np.array(image.shape[:2][::-1])[np.newaxis]
    kp_array = spatial_size * (kp_array + 1) / 2
    num_kp = kp_array.shape[0]
    colormap = plt.cm.get_cmap('gist_rainbow')
    for kp_ind, kp in enumerate(kp_array):
        rr, cc = circle(kp[1], kp[0], 2, shape=image.shape[:2])
        image[rr, cc] = np.array(colormap(kp_ind / num_kp))[:3]
    return image

def generate_video(config, inpainting_network, kp_detector, bg_predictor, dense_motion_network, pose_estimator, checkpoint, log_dir, dataset, if_dev=False):
    if checkpoint is not None:
        Logger.load_cpk(checkpoint, inpainting_network=inpainting_network, kp_detector=kp_detector,
                         bg_predictor=bg_predictor, dense_motion_network=dense_motion_network)
    else:
        raise AttributeError("Checkpoint should be specified for mode='reconstruction'.")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    inpainting_network.eval()
    kp_detector.eval()
    dense_motion_network.eval()
    if bg_predictor:
        bg_predictor.eval()

    torch.cuda.empty_cache()
    with torch.no_grad():
        # for frame_idx in range(len(length_list)):
        source_name = '/disk1/dataset/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-128x128px/test/31March_2010_Wednesday_tagesschau-999/images0001.png'
        driving_name = '/disk1/dataset/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-128x128px/test/31March_2010_Wednesday_tagesschau-999/images0022.png'
        source_img = io.imread(source_name)
        source_img = img_as_float32(source_img)
        source_video = np.moveaxis(source_img, 1, 0)
        source_video = source_video.reshape((-1,) + (128,128,3))
        source_video = np.moveaxis(source_video, 1, 2)

        driving_img = io.imread(driving_name)
        driving_img = img_as_float32(driving_img)
        driving_video = np.moveaxis(driving_img, 1, 0)
        driving_video = driving_video.reshape((-1,) + (128,128,3))
        driving_video = np.moveaxis(driving_video, 1, 2)
        source = torch.from_numpy(source_video).to(torch.float32).permute(0,3,1,2)
        driving = torch.from_numpy(driving_video).to(torch.float32).permute(0,3,1,2)
        if torch.cuda.is_available():
            source = source.cuda()
            driving = driving.cuda()
        if torch.cuda.is_available():
            source = source.cuda()
            driving = driving.cuda()
        kp_source = kp_detector(source)
        kp_driving = kp_detector(driving)
        pose_guidance = pose_estimator(driving)
        bg_params = None
        if bg_predictor:
            bg_params = bg_predictor(source, driving)
        dense_motion = dense_motion_network(source_image=source, kp_driving=kp_driving,
                                            kp_source=kp_source, bg_param = bg_params,
                                            dropout_flag = False)
        out = inpainting_network(source, dense_motion, pose_guidance)
        prediction = out['prediction'].data.cpu().numpy()
        # imageio.imsave('outputsss.png', (255 * prediction[0].transpose(1,2,0)).astype(np.uint8))

        # kp_source = kp_source['fg_kp'].data.cpu().numpy()
        # background = np.zeros((1, 128, 128,3))
        # source_array = draw_image_with_kp(background[0], kp_source[0])
        # imageio.imsave('source_kp.png', (255 * source_array).astype(np.uint8))

        deformation = dense_motion['deformation']
        deformation = deformation.permute(0, 3, 1, 2)
        deformation = interpolate(deformation, size=(128, 128), mode='bilinear', align_corners=True)
        deformation = deformation.permute(0, 2, 3, 1)
        deformation = deformation.data.cpu().numpy()
        x, y = np.meshgrid(np.arange(128), np.arange(128))
        grid = np.stack((x, y), axis=-1)/64-1
        deformation[0] = deformation[0] + grid

        # deformation = np.random.rand(1, 128, 128, 2)
        # flow = flow2rgb(deformation[0])
        flow = flow2img(deformation[0])
        imageio.imsave('flow_.png', (flow).astype(np.uint8))
        # # flow = interpolate(torch.from_numpy(flow).permute(2,0,1).unsqueeze(0), (128,128)).permute(0,2,3,1).numpy()
        # # imageio.imsave('flow.png', (255 * flow).astype(np.uint8))
        # flow_rgb = flow2rgb(deformation[0]*32)

        # # 使用 matplotlib 的 quiver 函数绘制箭头光流图
        # # 首先，我们需要从光流图中提取出横向和纵向的光流分量
        # flow_x = deformation[0][:, :, 0]
        # flow_y = deformation[0][:, :, 1]

        # # 创建一个网格，用于表示每个像素的位置
        # X, Y = np.meshgrid(np.arange(0, flow_x.shape[1], 1), np.arange(0, flow_x.shape[0], 1))

        # # 设置箭头的密度，以减少箭头的数量
        # skip = (slice(None, None, 2), slice(None, None, 2))

        # # 绘制箭头光流图
        # plt.figure(figsize=(10, 10))
        # plt.imshow(flow_rgb)
        # plt.quiver(X[skip], Y[skip], flow_x[skip], flow_y[skip], color='r', angles='xy', scale_units='xy', scale=1)
        # plt.axis('off')
        # plt.savefig('arrow_flow.png', bbox_inches='tight', pad_inches=0)

        # w, h = 32,32
        # x, y = np.meshgrid(np.arange(w), np.arange(h))
        # grid = np.stack((x, y), axis=-1)

        # grid_warped = grid + deformation[0]

        # # 将网格点转换为整数，以便在图像上绘制
        # grid_warped = grid_warped.astype(np.int32)

        # # 创建一个空白的BGR图像用于绘制网格
        # image = np.zeros((h, w, 3), dtype=np.uint8)

        # # 绘制原始网格
        # for i in range(h):
        #     for j in range(w):
        #         cv2.circle(image, (grid[i, j, 0], grid[i, j, 1]), 1, (0, 255, 0), -1)

        # # 绘制变形后的网格
        # for i in range(h):
        #     for j in range(w):
        #         cv2.circle(image, (grid_warped[i, j, 0], grid_warped[i, j, 1]), 1, (0, 0, 255), -1)
        # cv2.imwrite('grid.png', image)

        # kp_driving = kp_driving['fg_kp'].data.cpu().numpy()
        # driving_array = draw_image_with_kp(background[0], kp_driving[0])
        # imageio.imsave('driving_kp.png', (255 * driving_array).astype(np.uint8))
        # if 'occlusion_map' in out:
        #     for i in range(len(out['occlusion_map'])):
        #         occlusion_map = out['occlusion_map'][i].data.cpu().repeat(1, 3, 1, 1)
        #         occlusion_map = occlusion_map.numpy()
        #         occlusion_map = np.transpose(occlusion_map, [0, 2, 3, 1])
        #         # images.append(occlusion_map)
        #         imageio.imsave('occlusion_'+str(i)+'.png', (255 * occlusion_map[0]).astype(np.uint8))
        # for frame in range(prediction.shape[0]):
        #     image = np.transpose((255*prediction[frame]).astype(np.uint8),(1,2,0))
        #     folder_path = os.path.join(image_folder_path, x['name'][0])
        #     os.makedirs(folder_path, exist_ok=True)
        #     imageio.imsave(os.path.join(folder_path, 'images' + str(frame+frame_idx*length).zfill(4)+'.png'), (255 * prediction[frame].transpose(1,2,0)).astype(np.uint8))

