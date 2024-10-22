import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from logger import Logger
import numpy as np
import imageio
import math


def generate_video(config, inpainting_network, bg_predictor, dense_motion_network, checkpoint, log_dir, dataset,  if_dev=False):
    if checkpoint is not None:
        Logger.load_cpk(checkpoint, inpainting_network=inpainting_network,
                         bg_predictor=bg_predictor, dense_motion_network=dense_motion_network)
    else:
        raise AttributeError("Checkpoint should be specified for mode='reconstruction'.")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    inpainting_network.eval()
    dense_motion_network.eval()
    if bg_predictor:
        bg_predictor.eval()

    run_name = config['run_name']
    for it, x in tqdm(enumerate(dataloader)):
        torch.cuda.empty_cache()
        with torch.no_grad():
            if torch.cuda.is_available():
                x['video'] = x['video'].cuda()
            length = 12
            length_list = [length for i in range(math.ceil(x['video'].shape[2]/length))]
            if x['video'].shape[2] % length != 0:
                length_list[-1] = x['video'].shape[2] % length

            # log_dir = "/disk1/tongkai/dataset/fashion/"
            image_folder_path = os.path.join(log_dir, run_name, 'images')
            os.makedirs(image_folder_path, exist_ok=True)
            video_path = os.path.join(log_dir, run_name, 'video')
            os.makedirs(video_path, exist_ok=True)
            generate_video_path = os.path.join(video_path, x['name'][0]+'.mp4')
            writer = imageio.get_writer(generate_video_path, fps=30, format='mp4')

            for frame_idx in range(len(length_list)):
                source = torch.randn((length_list[frame_idx],x['video'].shape[1],x['video'].shape[3],x['video'].shape[4]))
                driving = torch.randn((length_list[frame_idx],x['video'].shape[1],x['video'].shape[3],x['video'].shape[4]))
                source_rdr = torch.randn((length_list[frame_idx],x['video'].shape[1],x['video'].shape[3],x['video'].shape[4]))
                driving_rdr = torch.randn((length_list[frame_idx],x['video'].shape[1],x['video'].shape[3],x['video'].shape[4]))
                source_depth = torch.randn((length_list[frame_idx],x['video_dp'].shape[1],x['video_dp'].shape[3],x['video_dp'].shape[4]))
                driving_depth = torch.randn((length_list[frame_idx],x['video_dp'].shape[1],x['video_dp'].shape[3],x['video_dp'].shape[4]))
                source_smpl = torch.randn((length_list[frame_idx],x['smpl_list'][0].shape[1],x['smpl_list'][0].shape[0]))
                driving_smpl = torch.randn((length_list[frame_idx],x['smpl_list'][0].shape[1],x['smpl_list'][0].shape[0]))

                for i in range(length_list[frame_idx]):
                    source[i] = x['video'][:, :, 0]
                    driving[i] = x['video'][:, :, frame_idx*length + i]
                    source_rdr[i] = x['video_rdr'][:, :, 0]
                    driving_rdr[i] = x['video_rdr'][:, :, frame_idx*length + i]
                    source_depth[i] = x['video_dp'][:,:,0]
                    driving_depth[i] = x['video_dp'][:, :, frame_idx*length + i]
                    source_smpl[i] = x['smpl_list'][0]
                    driving_smpl[i] = x['smpl_list'][frame_idx*length + i]

                if torch.cuda.is_available():
                    source = source.cuda()
                    driving = driving.cuda()
                    source_rdr = source_rdr.cuda()
                    driving_rdr = driving_rdr.cuda()
                    source_depth = source_depth.cuda()
                    driving_depth = driving_depth.cuda()
                    source_smpl = source_smpl.cuda()
                    driving_smpl = driving_smpl.cuda()

                bg_params = None
                if bg_predictor:
                    bg_params = bg_predictor(source, source_rdr, driving_rdr)

                source_rdr_in = torch.cat([source_rdr, source_depth], dim=1)
                driving_rdr_in = torch.cat([driving_rdr, driving_depth], dim=1)
                source_region_params = dense_motion_network(source_rdr_in, source_smpl)
                driving_region_params = dense_motion_network(driving_rdr_in, driving_smpl)

                out = inpainting_network(source, source_region_params=source_region_params,
                        #    driving_region_params=driving_region_params, bg=bg_predictor,
                           driving_region_params=driving_region_params, bg_params=bg_params,
                           driving_smpl=driving_smpl, source_smpl=source_smpl,
                           driving_smpl_rdr=driving_rdr, source_smpl_rdr=source_rdr,
                           driving_depth=driving_depth, source_depth=source_depth)
                prediction = out['prediction'].data.cpu().numpy()

                for frame in range(prediction.shape[0]):
                    image = np.transpose((255*prediction[frame]).astype(np.uint8),(1,2,0))
                    writer.append_data(image)

                    folder_path = os.path.join(image_folder_path, x['name'][0])
                    os.makedirs(folder_path, exist_ok=True)
                    imageio.imsave(os.path.join(folder_path, 'images' + str(frame+frame_idx*length).zfill(4)+'.png'), (255 * prediction[frame].transpose(1,2,0)).astype(np.uint8))

            writer.close()