

from PIL import Image

import numpy as np
import torch
from torchvision import transforms

tens_tens = transforms.ToTensor()


def prep_io(path, ch_inp, net_depth, noise_type):

    img_pil = Image.open(path).convert('RGB')

    img = tens_tens(img_pil).unsqueeze(0)
    img = make_even(img, net_depth)
    img_size = img.size()[2:]
    dummy = init_dummy(noise_type, img_size, ch_inp)

    return dummy, img


def prep_inpaint_mask(path, net_depth, same = False):
    mask = Image.open(path).convert('L')
    mask = tens_tens(mask).unsqueeze(0)
    if not same:
        mask = 1.0 - mask
    mask = make_even(mask, net_depth)

    return mask


def make_even(img, d): # Force image size to power of 2
    d = int(np.power(2, d))
    h, w = img.size()[2:]

    if h % 2 != 0:
        h -= 1
    if w % 2 != 0:
        w -= 1

    d_h = (h % d) // 2
    d_w = (w % d) // 2

    return img[:, :, d_h:h - d_h, d_w:w - d_w]

def init_dummy(noise_type, img_dims, ch_n, var=0.1):
    if noise_type == 'uniform':
        img = var * torch.rand((1, ch_n, img_dims[0], img_dims[1]))
    elif noise_type == 'normal':
        img = var * torch.randn((1, ch_n, img_dims[0], img_dims[1]))
    elif noise_type == 'mesh':
        assert ch_n == 2
        X, Y = np.meshgrid(np.arange(0, img_dims[1]) / float(img_dims[1] - 1),
                           np.arange(0, img_dims[0]) / float(img_dims[0] - 1))
        meshgrid = np.concatenate([X[None, :], Y[None, :]])
        img = torch.tensor(meshgrid).unsqueeze(0).type(torch.float)

    elif noise_type == 'special':
        X, Y = np.meshgrid(np.arange(0, img_dims[1]) / float(img_dims[1] - 1),
                           np.arange(0, img_dims[0]) / float(img_dims[0] - 1))
        meshgrid = np.concatenate([X[None, :], Y[None, :]])
        img = torch.tensor(meshgrid).unsqueeze(0).type(torch.float)
        img = torch.cat((img, torch.ones((1, 1, img_dims[0], img_dims[1]))), dim=1)
    return img


def add_noise(img_tensor, mean=0.0, std=0.1):
    img_tensor += torch.randn(img_tensor.size()) * std + mean
    return torch.clip(img_tensor, 0.0, 1.0).type(torch.float)
