import os
import argparse
import torch
import subprocess
import glob
from tqdm import tqdm

from utils import load_txt_file, mkdirs
import render_process

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", type=str, default="/disk1/dataset/FashionVideo", help="the root directory of dataset.")
    parser.add_argument("--output_dir", type=str, default="/disk1/dataset/FashionVideo", help="the root directory of dataset.")
    # parser.add_argument("--gpu_id", type=str, default="0", help="the gpu ids.")
    parser.add_argument("--workers", type=int, default=16, help="numbers of workers")
    parser.add_argument("--batch_size", type=int, default=320, help="numbers of batch size for SMPL")
    parser.add_argument("--image_size", type=int, default=256, help="the image size.")
    args = parser.parse_args()

    # os.environ["CUDA_DEVICES_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    # device = torch.device("cuda:0")
    device = torch.device("cuda")

    stages = ['train', 'test']

    # train = load_txt_file(os.path.join(args.video_dir, 'fashion_train.txt'))
    # test = load_txt_file(os.path.join(args.video_dir, 'fashion_test.txt'))
    train = [item for item in os.listdir(os.path.join(args.video_dir, 'train')) if os.path.isdir(os.path.join(os.path.join(args.video_dir, 'train'), item))]
    test = [item for item in os.listdir(os.path.join(args.video_dir, 'test')) if os.path.isdir(os.path.join(os.path.join(args.video_dir, 'test'), item))]
    train_names = [['train', os.path.basename(each)] for each in train]
    test_names = [['test', os.path.basename(each)] for each in test]
    names = train_names + test_names

    for stage, name in tqdm(names):
        # save_dir_visualization_frames = os.path.join(args.output_dir, stage, name, 'depth')
        save_dir_visualization_frames = os.path.join(args.output_dir, stage, name, 'normal_new')
        # print(save_dir_visualization_frames)
        mkdirs(save_dir_visualization_frames)

        frames_dir = os.path.join(args.video_dir, stage, name, 'frames')
        frames = sorted(glob.glob(os.path.join(frames_dir, '*.png')))
        # print(frames)
        render_process.render_process(frames, save_dir_visualization_frames,
                           args.image_size, device, args.workers, args.batch_size)

