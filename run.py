import matplotlib
matplotlib.use('Agg')
import torch
import random
import numpy as np

seed = 54321

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

import os
import yaml
from argparse import ArgumentParser
from time import gmtime, strftime
from shutil import copy
from data.frames_dataset import FramesDataset

from modules.generator import Generator
from modules.bg_motion_predictor import BGMotionPredictor
from modules.region_predictor import RegionPredictor
import torch
from train import train
from generate_video import generate_video
import os
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", default="config/lsa.yaml", help="path to config")
    parser.add_argument("--mode", default="train", choices=["train", "generate_video"])
    parser.add_argument("--log_dir", default='log', help="path to log into")
    parser.add_argument("--run_name", default=None, help="wandb run name")
    parser.add_argument("--dataset", default='Fashion_new', help="wandb project name")
    parser.add_argument("--checkpoint", default=None, help="path to checkpoint to restore")
    parser.add_argument("--device_ids", default="0",
                        help="Names of the devices comma separated.")
    parser.add_argument("--if_dev", default=False, help="if dev")

    opt = parser.parse_args()
    with open(opt.config) as f:
        config = yaml.safe_load(f)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.device_ids)

    if opt.run_name == None:
        raise RuntimeError("please input the run name!!")
    config['run_name'] = opt.run_name
    config['dataset'] = opt.dataset

    if opt.checkpoint is not None:
        log_dir = os.path.join(*os.path.split(opt.checkpoint)[:-1])
    else:
        log_dir = os.path.join(opt.log_dir, opt.run_name)
        if os.path.exists(log_dir):
            log_dir += '-' + strftime("%d_%m_%y_%H.%M.%S", gmtime())

    inpainting = Generator(num_regions=config['model_params']['common_params']['num_regions'],
                          num_channels=config['model_params']['common_params']['num_channels'],
                          revert_axis_swap=config['model_params']['common_params']['revert_axis_swap'],
                          **config['model_params']['generator_params'])

    if torch.cuda.is_available():
        inpainting.cuda()

    region_predictor = RegionPredictor(num_regions=config['model_params']['common_params']['num_regions'],
                                       num_channels=config['model_params']['common_params']['num_channels'],
                                       estimate_affine=config['model_params']['common_params']['estimate_affine'],
                                              **config['model_params']['region_predictor_params'])

    if torch.cuda.is_available():
        region_predictor.cuda()

    bg_predictor = None
    if (config['model_params']['common_params']['bg']):
        bg_predictor = BGMotionPredictor(num_channels=config['model_params']['common_params']['num_channels'],
                                     **config['model_params']['bg_predictor_params'])
        if torch.cuda.is_available():
            bg_predictor.cuda()

    dataset = FramesDataset(is_train=(opt.mode.startswith('train')), **config['dataset_params'])
    test_dataset = FramesDataset(is_train=False, **config['dataset_params'])

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(os.path.join(log_dir, os.path.basename(opt.config))):
        copy(opt.config, log_dir)

    if opt.mode == 'train':
        print("Training...")
        train(config, inpainting, bg_predictor, region_predictor, opt.checkpoint, log_dir, dataset)
    elif opt.mode == 'generate_video':
        log_dir = opt.log_dir
        print("Generating")
        generate_video(config, inpainting, bg_predictor, region_predictor, opt.checkpoint, log_dir, test_dataset)