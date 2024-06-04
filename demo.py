import matplotlib
matplotlib.use('Agg')
import sys
import yaml
from argparse import ArgumentParser
from tqdm import tqdm
from scipy.spatial import ConvexHull
import numpy as np
import imageio
from skimage.transform import resize
from skimage import img_as_ubyte
import torch
from modules.generator import Generator as InpaintingNetwork
# from modules.keypoint_detector import KPDetector
from modules.region_predictor import RegionPredictor as DenseMotionNetwork
from modules.bg_motion_predictor import BGMotionPredictor
# from modules.avd_network import AVDNetwork

if sys.version_info[0] < 3:
    raise Exception("You must use Python 3 or higher. Recommended version is Python 3.9")

def relative_kp(kp_source, kp_driving, kp_driving_initial):

    source_area = ConvexHull(kp_source['fg_kp'][0].data.cpu().numpy()).volume
    driving_area = ConvexHull(kp_driving_initial['fg_kp'][0].data.cpu().numpy()).volume
    adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)

    kp_new = {k: v for k, v in kp_driving.items()}

    kp_value_diff = (kp_driving['fg_kp'] - kp_driving_initial['fg_kp'])
    kp_value_diff *= adapt_movement_scale
    kp_new['fg_kp'] = kp_value_diff + kp_source['fg_kp']

    return kp_new

def load_checkpoints(config_path, checkpoint_path, device):
    with open(config_path) as f:
        config = yaml.full_load(f)

    inpainting = InpaintingNetwork(num_regions=config['model_params']['common_params']['num_regions'],
                          num_channels=config['model_params']['common_params']['num_channels'],
                          revert_axis_swap=config['model_params']['common_params']['revert_axis_swap'],
                          **config['model_params']['generator_params'])
    # kp_detector = KPDetector(**config['model_params']['common_params'])
    dense_motion_network = DenseMotionNetwork(num_regions=config['model_params']['common_params']['num_regions'],
                                       num_channels=config['model_params']['common_params']['num_channels'],
                                       estimate_affine=config['model_params']['common_params']['estimate_affine'],
                                              **config['model_params']['region_predictor_params'])
    bg_predictor = BGMotionPredictor(num_channels=config['model_params']['common_params']['num_channels'],
                                     **config['model_params']['bg_predictor_params'])
    
    dense_motion_network.to(device)
    inpainting.to(device)
    bg_predictor.to(device)
       
    checkpoint = torch.load(checkpoint_path, map_location=device)
 
    inpainting.load_state_dict(checkpoint['inpainting_network'])
    dense_motion_network.load_state_dict(checkpoint['dense_motion_network'])
    bg_predictor.load_state_dict(checkpoint['bg_predictor'])
    
    inpainting.eval()
    dense_motion_network.eval()
    bg_predictor.eval()
    
    return inpainting, dense_motion_network, bg_predictor


def make_animation(source_image, driving_video, inpainting_network, dense_motion_network, bg_predictor, device, mode = 'relative'):
    assert mode in ['standard', 'relative', 'avd']
    with torch.no_grad():
        predictions = []
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        source = source.to(device)
        driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3).to(device)
        source_region_params = dense_motion_network(source)
                        

        for frame_idx in tqdm(range(driving.shape[2])):
            driving_frame = driving[:, :, frame_idx]
            driving_frame = driving_frame.to(device)
            driving_region_params = dense_motion_network(driving_frame)
            bg_params = bg_predictor(source, driving_frame)

            out = inpainting_network(source, source_region_params=source_region_params,
                                   driving_region_params=driving_region_params, bg_params=bg_params)

            predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
    return predictions

def inference_img(source_image, driving_image, inpainting_network, dense_motion_network, bg_predictor, device):
    with torch.no_grad():
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2).to(device)
        driving = torch.tensor(driving_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2).to(device)
        source_region_params = dense_motion_network(source)
        driving_region_params = dense_motion_network(driving)
        bg_params = bg_predictor(source, driving)
        out = inpainting_network(source, source_region_params=source_region_params,
                                   driving_region_params=driving_region_params, bg_params=bg_params)

    return out

def find_best_frame(source, driving, cpu):
    import face_alignment

    def normalize_kp(kp):
        kp = kp - kp.mean(axis=0, keepdims=True)
        area = ConvexHull(kp[:, :2]).volume
        area = np.sqrt(area)
        kp[:, :2] = kp[:, :2] / area
        return kp

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True,
                                      device= 'cpu' if cpu else 'cuda')
    kp_source = fa.get_landmarks(255 * source)[0]
    kp_source = normalize_kp(kp_source)
    norm  = float('inf')
    frame_num = 0
    for i, image in tqdm(enumerate(driving)):
        try:
            kp_driving = fa.get_landmarks(255 * image)[0]
            kp_driving = normalize_kp(kp_driving)
            new_norm = (np.abs(kp_source - kp_driving) ** 2).sum()
            if new_norm < norm:
                norm = new_norm
                frame_num = i
        except:
            pass
    return frame_num


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--checkpoint", default='checkpoints/vox.pth.tar', help="path to checkpoint to restore")

    parser.add_argument("--source_image", default='./assets/source.png', help="path to source image")
    parser.add_argument("--driving_video", default='./assets/driving.mp4', help="path to driving video")
    parser.add_argument("--result_video", default='./result.mp4', help="path to output")
    
    parser.add_argument("--img_shape", default="256,256", type=lambda x: list(map(int, x.split(','))),
                        help='Shape of image, that the model was trained on.')
    
    parser.add_argument("--mode", default='relative', choices=['standard', 'relative', 'avd'], help="Animate mode: ['standard', 'relative', 'avd'], when use the relative mode to animate a face, use '--find_best_frame' can get better quality result")
    
    parser.add_argument("--find_best_frame", dest="find_best_frame", action="store_true", 
                        help="Generate from the frame that is the most alligned with source. (Only for faces, requires face_aligment lib)")

    parser.add_argument("--cpu", dest="cpu", action="store_true", help="cpu mode.")

    opt = parser.parse_args()

    source_image = imageio.imread(opt.source_image)
    reader = imageio.get_reader(opt.driving_video)
    fps = reader.get_meta_data()['fps']
    driving_video = []
    try:
        for im in reader:
            driving_video.append(im)
    except RuntimeError:
        pass
    reader.close()
    
    if opt.cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
    
    source_image = resize(source_image, opt.img_shape)[..., :3]
    driving_video = [resize(frame, opt.img_shape)[..., :3] for frame in driving_video]
    inpainting, kp_detector, dense_motion_network, avd_network = load_checkpoints(config_path = opt.config, checkpoint_path = opt.checkpoint, device = device)
 
    if opt.find_best_frame:
        i = find_best_frame(source_image, driving_video, opt.cpu)
        print ("Best frame: " + str(i))
        driving_forward = driving_video[i:]
        driving_backward = driving_video[:(i+1)][::-1]
        predictions_forward = make_animation(source_image, driving_forward, inpainting, kp_detector, dense_motion_network, avd_network, device = device, mode = opt.mode)
        predictions_backward = make_animation(source_image, driving_backward, inpainting, kp_detector, dense_motion_network, avd_network, device = device, mode = opt.mode)
        predictions = predictions_backward[::-1] + predictions_forward[1:]
    else:
        predictions = make_animation(source_image, driving_video, inpainting, kp_detector, dense_motion_network, avd_network, device = device, mode = opt.mode)
    
    imageio.mimsave(opt.result_video, [img_as_ubyte(frame) for frame in predictions], fps=fps)

