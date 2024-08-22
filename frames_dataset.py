import os
import re
from skimage import io, img_as_float32
from skimage.color import gray2rgb
from sklearn.model_selection import train_test_split
from imageio import mimread
from skimage.transform import resize
import numpy as np
from torch.utils.data import Dataset
from augmentation import AllAugmentationTransform
import glob
from functools import partial
import json


def read_video(name, frame_shape):
    """
    Read video which can be:
      - an image of concatenated frames
      - '.mp4' and'.gif'
      - folder with videos
    """

    if os.path.isdir(name):
        frames = sorted(os.listdir(name))
        num_frames = len(frames)
        video_array = np.array(
            [img_as_float32(io.imread(os.path.join(name, frames[idx]))) for idx in range(num_frames)])
    elif name.lower().endswith('.png') or name.lower().endswith('.jpg'):
        image = io.imread(name)

        if len(image.shape) == 2 or image.shape[2] == 1:
            image = gray2rgb(image)

        if image.shape[2] == 4:
            image = image[..., :3]

        image = img_as_float32(image)

        video_array = np.moveaxis(image, 1, 0)

        video_array = video_array.reshape((-1,) + frame_shape)
        video_array = np.moveaxis(video_array, 1, 2)
    elif name.lower().endswith('.gif') or name.lower().endswith('.mp4') or name.lower().endswith('.mov'):
        video = mimread(name)
        if len(video[0].shape) == 2:
            video = [gray2rgb(frame) for frame in video]
        if frame_shape is not None:
            video = np.array([resize(frame, frame_shape) for frame in video])
        video = np.array(video)
        if video.shape[-1] == 4:
            video = video[..., :3]
        video_array = img_as_float32(video)
    else:
        raise Exception("Unknown file extensions  %s" % name)

    return video_array


def read_json2np(json_path):
    with open(json_path, 'r') as f:
        json_file = json.load(f)
    cam = np.array(json_file['cam'], dtype='float32').reshape(-1,1)
    pose = np.array(json_file['pose_theta'], dtype='float32').reshape(-1,1)
    shape = np.array(json_file['shape_beta'], dtype='float32').reshape(-1,1)

    return np.concatenate((cam, pose, shape), axis=0)


class FramesDataset(Dataset):
    """
    origin, smpl, and its json file
    The struct of xxx is
    xxx/
        frames/  origin video frames
        rendered/   smpl video frames
        kptsmpls/   keypoints and smpl json frames
    """

    def __init__(self, root_dir, frame_shape=(256, 256, 3), id_sampling=False, is_train=True,
                 random_seed=1234, pairs_list=None, augmentation_params=None, is_dev=False):
        self.root_dir = root_dir
        self.videos = os.listdir(root_dir)
        self.frame_shape = frame_shape
        print(self.frame_shape)
        self.pairs_list = pairs_list
        self.id_sampling = id_sampling

        if os.path.exists(os.path.join(root_dir, 'train')):
            assert os.path.exists(os.path.join(root_dir, 'test'))
            print("Use predefined train-test split.")
            if id_sampling:
                train_videos = {os.path.basename(video).split('#')[0] for video in
                                os.listdir(os.path.join(root_dir, 'train'))}
                train_videos = list(train_videos)
            else:
                train_videos = os.listdir(os.path.join(root_dir, 'train'))
            test_videos = os.listdir(os.path.join(root_dir, 'test'))
            if is_dev:
                dev_videos = os.listdir(os.path.join(root_dir, 'dev'))
            self.root_dir = os.path.join(self.root_dir, 'train' if is_train else 'test')
            if is_dev:
                self.root_dir = os.path.join(root_dir, 'dev')
        else:
            print("Use random train-test split.")
            train_videos, test_videos = train_test_split(self.videos, random_state=random_seed, test_size=0.2)

        if is_train:
            self.videos = train_videos
        else:
            self.videos = test_videos
        if is_dev:
            self.videos = dev_videos

        self.is_train = is_train
        self.videos.sort()

        if self.is_train:
            self.transform = AllAugmentationTransform(**augmentation_params)
        else:
            self.transform = None

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):

        if self.is_train and self.id_sampling:
            name = self.videos[idx]
            path = np.random.choice(glob.glob(os.path.join(self.root_dir, name + '*.mp4')))
        else:
            name = self.videos[idx]
            path = os.path.join(self.root_dir, name)

        video_name = os.path.basename(path)
        video_path = os.path.join(path, 'frames')
        rendered_path = os.path.join(path, 'normal')
        json_path = os.path.join(path, 'kptsmpls')
        depth_path = os.path.join(path, 'depth')
        if self.is_train:
            frames = os.listdir(video_path)
            frames.sort()
            num_frames = len(frames)
            frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2))

            if self.frame_shape is not None:
                resize_fn = partial(resize, output_shape=self.frame_shape)
            else:
                resize_fn = img_as_float32

            if type(frames[0]) is bytes:
                video_array = [resize_fn(io.imread(os.path.join(video_path, frames[idx].decode('utf-8')))) for idx in frame_idx]
                rendered_array = [resize_fn(io.imread(os.path.join(rendered_path, frames[idx].decode('utf-8')))) for idx in frame_idx]
                depth_array = [resize_fn(io.imread(os.path.join(depth_path, frames[idx].decode('utf-8')), as_gray=True)) for idx in frame_idx]
                smpl_list = [read_json2np(os.path.join(json_path, frames[idx].decode('utf-8'))) for idx in frame_idx]
            else:
                video_array = [resize_fn(io.imread(os.path.join(video_path, frames[idx]))) for idx in frame_idx]
                rendered_array = [resize_fn(io.imread(os.path.join(rendered_path, frames[idx]))) for idx in frame_idx]
                depth_array = [resize_fn(io.imread(os.path.join(depth_path, frames[idx]), as_gray=True)) for idx in frame_idx]
                smpl_list = [read_json2np(os.path.join(json_path, frames[idx]).replace('.png', '.json')) for idx in frame_idx]
        else:
            video_array = read_video(video_path, frame_shape=self.frame_shape)
            rendered_array = read_video(rendered_path, frame_shape=self.frame_shape)
            depth_array = read_video(depth_path, frame_shape=self.frame_shape, as_gray=True)


            num_frames = len(video_array)
            frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2)) if self.is_train else range(
                num_frames)
            video_array = video_array[frame_idx]
            rendered_array = rendered_array[frame_idx]
            depth_array = depth_array[frame_idx]

            frames = sorted(os.listdir(json_path))
            smpl_list = [read_json2np(os.path.join(json_path, frames[idx])) for idx in frame_idx]



        if self.transform is not None:
            video_array = self.transform(video_array)

        out = {}
        if self.is_train:
            source = np.array(video_array[0], dtype='float32')
            driving = np.array(video_array[1], dtype='float32')
            source_rdr = np.array(rendered_array[0], dtype='float32')
            driving_rdr = np.array(rendered_array[1], dtype='float32')
            source_dp = np.array(depth_array[0], dtype='float32')
            driving_dp = np.array(depth_array[1], dtype='float32')
            source_smpl = smpl_list[0]
            driving_smpl = smpl_list[1]

            out['driving'] = driving.transpose((2, 0, 1))
            out['source'] = source.transpose((2, 0, 1))
            out['driving_rdr'] = driving_rdr.transpose((2, 0, 1))
            out['source_rdr'] = source_rdr.transpose((2, 0, 1))
            out['driving_dp'] = np.expand_dims(driving_dp, axis=0)
            out['source_dp'] = np.expand_dims(source_dp, axis=0)
            out['driving_smpl'] = driving_smpl
            out['source_smpl'] = source_smpl
        else:
            video = np.array(video_array, dtype='float32')
            out['video'] = video.transpose((3, 0, 1, 2))
            video_rdr = np.array(rendered_array, dtype='float32')
            out['video_rdr'] = video_rdr.transpose((3, 0, 1, 2))
            video_dp = np.array(depth_array, dtype='float32')
            video_dp = np.expand_dims(video_dp, axis=0)
            out['video_dp'] = video_dp
            out['smpl_list'] = smpl_list

        out['name'] = video_name
        return out


class DatasetRepeater(Dataset):
    """
    Pass several times over the same dataset for better i/o performance
    """

    def __init__(self, dataset, num_repeats=100):
        self.dataset = dataset
        self.num_repeats = num_repeats

    def __len__(self):
        return self.num_repeats * self.dataset.__len__()

    def __getitem__(self, idx):
        return self.dataset[idx % self.dataset.__len__()]

