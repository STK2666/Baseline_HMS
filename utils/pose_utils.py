import numpy as np
import torch
import sys
sys.path.append('/ssd5/tongkai/Baseline_HMS')
from modules.util import kpt2heatmap
from SMPLDataset.human_digitalizer.bodynets import SMPL

MISSING_VALUE = -1


def orthographic_proj_withz_idrot(X, cam, offset_z=0.):
    """
    X: B x N x 3
    cam: B x 3: [sc, tx, ty]
    No rotation!
    Orth preserving the z.
    sc * ( x + [tx; ty])
    as in HMR..
    """
    scale = cam[:, 0].contiguous().view(-1, 1, 1)
    trans = cam[:, 1:3].contiguous().view(cam.size(0), 1, -1)

    # proj = scale * X
    proj = X

    proj_xy = scale * (proj[:, :, :2] + trans)
    # proj_z = proj[:, :, 2, None] + offset_z

    return proj_xy


def smpl2kpts(smpl_params):
    smpl_model = SMPL('./SMPLDataset/checkpoints/smpl_model.pkl').eval().to(smpl_params.device)
    smpl_model.requires_grad_(False)
    cam = smpl_params[:,0:3].contiguous()
    pose_theta = smpl_params[:,3:75].contiguous()
    shape_beta = smpl_params[:,75:].contiguous()
    joints = smpl_model(beta=shape_beta, theta=pose_theta)

    keypoints = orthographic_proj_withz_idrot(joints, cam)

    keypoints = (keypoints + 1) / 2

    # N,K,2 (-1,1)
    return keypoints


def visualize_keypoints(keypoints, image_size=(512, 512), output_path='keypoints.png'):
    import cv2
    import imageio
    # Create an empty white image
    # image = np.ones((image_size[0], image_size[1], 3), dtype=np.uint8) * 255
    image = cv2.imread('/ssd5/tongkai/FashionVideo/test/91EfnBTEE2S.mp4/frames/frame_00000.png')
    image = np.ones_like(image) * 255

    # Rescale keypoints to image coordinates
    keypoints = keypoints[0].cpu().numpy()  # Assuming batch size is 1
    keypoints = keypoints * np.array([image_size[1], image_size[0]])

    for kp in keypoints:
        x, y = int(kp[0]), int(kp[1])
        cv2.circle(image, (x, y), radius=2, color=(0, 0, 255), thickness=-1)

    # Save image using imageio
    imageio.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def save_combined_heatmaps(heatmaps, n=0):
    import imageio
    combined_heatmap = heatmaps[n].sum(dim=0)  # 叠加所有K个热图
    combined_heatmap = combined_heatmap / combined_heatmap.max()  # 归一化到 [0, 1]
    combined_heatmap = (combined_heatmap * 255).cpu().numpy().astype(np.uint8)  # 转换为uint8

    filename = f'/ssd5/tongkai/Baseline_HMS/utils/combined_heatmap_n{n}.png'
    imageio.imwrite(filename, combined_heatmap)
    print(f"Saved {filename}")


if __name__ == '__main__':
    # cam = np.array([1.0183262825012207, -0.005766093730926514, 0.10632294416427612])
    # pose_theta = np.array([3.0322132110595703, 0.006083549931645393, 0.19405201077461243,
    #                        -0.14107580482959747, -0.08731464296579361, -0.08470478653907776,
    #                        -0.20121519267559052, -0.003623948199674487, 0.033302828669548035,
    #                        0.17076468467712402, 0.031040843576192856, 0.025391053408384323,
    #                        0.3966883718967438, -0.13841724395751953, -0.03642634302377701,
    #                        0.46564781665802, -0.010276616550981998, 0.05136444792151451,
    #                        0.009724756702780724, 0.03637612238526344, 0.001174240023829043,
    #                        0.1854560226202011, 0.11310181766748428, -0.05647023022174835,
    #                        0.08585157245397568, -0.19483143091201782, 0.13707181811332703,
    #                        0.0857841894030571, 0.007797864265739918, -0.0007844266947358847,
    #                        -0.329008549451828, 0.17672859132289886, 0.12035874277353287,
    #                        -0.059008706361055374, 0.1518184393644333, -0.27163490653038025,
    #                        -0.04586087912321091, -0.05503547936677933, 0.07269243150949478,
    #                        0.049698278307914734, 0.10340209305286407, -0.41512563824653625,
    #                        -0.024291379377245903, 0.09605257213115692, 0.3944506347179413,
    #                        0.2661330997943878, -0.1218220591545105, -0.03004622459411621,
    #                        0.1757097691297531, -0.24193714559078217, -1.0886828899383545,
    #                        0.16731548309326172, 0.35544976592063904, 0.9983264803886414,
    #                        0.2247839719057083, -0.5399426221847534, 0.10894361883401871,
    #                        -0.042721278965473175, 0.4862653315067291, -0.15916472673416138,
    #                        -0.024443838745355606, -0.003774648765102029, -0.08877009153366089,
    #                        -0.02254605107009411, 0.017699984833598137, 0.06533411890268326,
    #                        -0.20381803810596466, -0.06606652587652206, -0.18190881609916687,
    #                        -0.10058648884296417, 0.09969579428434372, 0.1905435025691986])
    # # pose_theta = pose_theta.reshape(-1, 3)
    # shape_beta = np.array([0.24388732016086578, 1.1266093254089355, 0.9954346418380737,
    #                        1.8308491706848145, -0.5458313822746277, 0.06264655292034149,
    #                        -0.28186672925949097, 0.4273015558719635, 0.22667549550533295,
    #                        -0.17520955204963684])
    cam = np.array([1.0357822179794312, 0.007449328899383545, 0.15112167596817017])
    pose_theta = np.array([
        3.1203060150146484,
        0.03368666395545006,
        0.3668786287307739,
        -0.14121751487255096,
        -0.06567519158124924,
        -0.04971429333090782,
        -0.13083617389202118,
        -0.05400417000055313,
        -0.0659375861287117,
        0.09829283505678177,
        0.03655988350510597,
        0.036067280918359756,
        0.4604203701019287,
        -0.21292154490947723,
        -0.06980738043785095,
        0.5318703055381775,
        -0.005865656770765781,
        0.05803794413805008,
        0.00028130586724728346,
        0.02194575034081936,
        0.018371403217315674,
        0.069588802754879,
        0.0840354934334755,
        -0.07221314311027527,
        0.037248846143484116,
        -0.21362990140914917,
        0.16419857740402222,
        0.0807264894247055,
        -0.000703414436429739,
        0.010199577547609806,
        -0.3412783741950989,
        0.14882220327854156,
        0.15912246704101562,
        -0.10662747919559479,
        0.13853058218955994,
        -0.3213016390800476,
        -0.17761577665805817,
        -0.14156900346279144,
        0.10821137577295303,
        0.05547407269477844,
        0.1436978280544281,
        -0.37031838297843933,
        -0.046287018805742264,
        0.06743593513965607,
        0.3515608012676239,
        0.16287338733673096,
        -0.1997104287147522,
        0.009822462685406208,
        0.2150481790304184,
        -0.14787834882736206,
        -1.0826390981674194,
        0.1608041226863861,
        0.2939387559890747,
        0.9844391345977783,
        0.3354889452457428,
        -0.5558665990829468,
        0.07324878871440887,
        -0.003428824245929718,
        0.5561004281044006,
        -0.1567717045545578,
        -0.07810278236865997,
        -0.03759455680847168,
        -0.06000906229019165,
        -0.022617070004343987,
        0.029735760763287544,
        0.03591436147689819,
        -0.21338750422000885,
        -0.0521605908870697,
        -0.19940368831157684,
        -0.10090349614620209,
        0.09673436731100082,
        0.18085475265979767
    ])
    shape_beta = np.array([
        0.16810624301433563,
        0.6991161108016968,
        1.1932835578918457,
        2.4073994159698486,
        -0.2699187695980072,
        0.23886768519878387,
        -0.3054838180541992,
        0.39721229672431946,
        0.18714964389801025,
        -0.1794012486934662
    ])
    smpl_params = np.concatenate((cam, pose_theta, shape_beta), axis=0)
    smpl_params = torch.tensor(smpl_params, dtype=torch.float32).unsqueeze(0)
    keypoints = smpl2kpts(smpl_params)
    print(keypoints)
    print(keypoints.shape)
    visualize_keypoints(keypoints, image_size=(256, 256), output_path='/ssd5/tongkai/Baseline_HMS/utils/keypoints.png')

    pose_joints = keypoints * 256
    im_size = (256, 256)
    map = kpt2heatmap(pose_joints, im_size, sigma=3.0)
    print(map)
    print(map.shape)

    save_combined_heatmaps(map)