import torch
import torch.nn as nn
import numpy as np
from PIL import Image

from torchvision import transforms

from skimage.feature import canny
from skimage.color import gray2rgb, rgb2gray


def tensor_to_image():

    return transforms.ToPILImage()


def image_to_tensor():

    return transforms.ToTensor()


def image_to_edge(image, sigma):

    gray_image = rgb2gray(np.array(tensor_to_image()(image)))
    edge = image_to_tensor()(Image.fromarray(canny(gray_image, sigma=sigma)))
    gray_image = image_to_tensor()(Image.fromarray(gray_image))

    return edge, gray_image


def image_to_edge_tensor(image_tensor, sigma=2.0):
    batch_size, channels, height, width = image_tensor.shape
    edges = []
    gray_images = []

    for i in range(batch_size):
        single_image_tensor = image_tensor[i]

        np_image = single_image_tensor.permute(1, 2, 0).cpu().detach().numpy()
        gray_image = rgb2gray(np_image)

        edge_tensor = image_to_tensor()(Image.fromarray(canny(gray_image, sigma=sigma)))

        gray_image_tensor = image_to_tensor()(Image.fromarray(gray_image))

        edges.append(edge_tensor)
        gray_images.append(gray_image_tensor)

    edges_tensor = torch.stack(edges).to(image_tensor.device)
    gray_images_tensor = torch.stack(gray_images).to(image_tensor.device)

    return edges_tensor, gray_images_tensor


def weights_init(init_type='normal', gain=0.02):

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):

            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=gain)

            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, gain)
            nn.init.constant_(m.bias.data, 0.0)

    return init_func


def gram_matrix(feat):

    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)

    return gram


def spectral_norm(module, mode=True):

    if mode:
        return nn.utils.spectral_norm(module)

    return module


def postprocess(x):

    x = (x + 1.) / 2.
    x.clamp_(0, 1)
    return x


def extract_patches(x, kernel_size=3, stride=1):

    if kernel_size != 1:
        x = nn.ZeroPad2d(1)(x)
    x = x.permute(0, 2, 3, 1)
    x = x.unfold(1, kernel_size, stride).unfold(2, kernel_size, stride)
    return x.contiguous()


def requires_grad(model, flag=True):

    for p in model.parameters():
        p.requires_grad = flag


def sample_data(loader):

    while True:
        for batch in loader:
            yield batch


# def discriminator_loss_func(real_pred, fake_pred, real_pred_edge, fake_pred_edge, edge):

#     criterion = nn.BCELoss()

#     real_target = torch.tensor(1.0).expand_as(real_pred)
#     fake_target = torch.tensor(0.0).expand_as(fake_pred)
#     if torch.cuda.is_available():
#         real_target = real_target.cuda()
#         fake_target = fake_target.cuda()

#     loss_adversarial = criterion(real_pred, real_target) + criterion(fake_pred, fake_target) + \
#                     criterion(real_pred_edge, edge) + criterion(fake_pred_edge, edge)

#     return {
#         'loss_adversarial': loss_adversarial.mean()
#     }


def discriminator_loss_func(real_pred, fake_pred, weight=1.0):

    criterion = nn.BCELoss()

    real_target = torch.tensor(1.0).expand_as(real_pred)
    fake_target = torch.tensor(0.0).expand_as(fake_pred)
    if torch.cuda.is_available():
        real_target = real_target.cuda()
        fake_target = fake_target.cuda()

    loss_adversarial = criterion(real_pred, real_target) + criterion(fake_pred, fake_target)

    return {
        'loss_adversarial': loss_adversarial.mean() * weight
    }