from torchvision import transforms

from cv_utils import *

tens_norm = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(0.5, 0.5)])
tens_tens = transforms.ToTensor()


def prep_io(path, ch_inp, ch_out, net_depth, noise_type, saliecity=False):
    if ch_out == 3:
        img_pil = open_image(path, tp='pil_rgb')
    else:
        img_pil = open_image(path, tp='pil_gray')

    img = image2Tensor(img_pil)
    img = make_even(img, net_depth)
    img_size = img.size()[2:]
    dummy = init_dummy(noise_type, img_size, ch_inp)

    if saliecity:
        img_np = pil2np(img_pil)
        slcy = saliency_detection(img_np)
        slcy = image2Tensor(slcy, norm_type='none')

        return dummy, img, slcy

    return dummy, img


def prep_inpaint_mask(path, net_depth, same = False):
    mask = Image.open(path).convert('L')
    mask = image2Tensor(mask, norm_type='scale')
    if not same:
        mask = 1.0 - mask
    mask = make_even(mask, net_depth)

    return mask


def make_even(img, d):
    d = int(np.power(2, d))
    h, w = img.size()[2:]

    if h % 2 != 0:
        h -= 1
    if w % 2 != 0:
        w -= 1

    d_h = (h % d) // 2
    d_w = (w % d) // 2

    return img[:, :, d_h:h - d_h, d_w:w - d_w]


def image2Tensor(image, norm_type='scale'):  # scale, mean
    if norm_type == 'scale':
        image = tens_tens(image)
    elif norm_type == 'mean':
        image = tens_norm(image)
    else:
        image = torch.tensor(image).unsqueeze(0)
    return image.unsqueeze(0)


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


def get_dft(image_torch):
    image_np = np.array(image_torch)[0, 0]
    dft = img2dft(image_np, False)
    dft = np.transpose(dft, (2, 0, 1))
    #diff = dft.max() - dft.min()
    dft = dft / 1000.0
    dft = torch.tensor(dft)
    return dft.unsqueeze(0)


def get_idft(dft_torch):
    dft_np = np.array(dft_torch.cpu())[0]
    dft_np = np.transpose(dft_np, (1, 2, 0))
    img = dft2img(dft_np)
    img = torch.tensor(img).unsqueeze(0) * 1000.0
    return img.unsqueeze(0)

def magnitude(dft_torch):
    mag = torch.pow(dft_torch, 2).sqrt().sum(dim = 1)
    mag = 20*torch.log(mag + 1)
    mag_d = mag.max() - mag.min()
    mag = (mag -  mag.min()) / mag_d
    return mag

def add_noise(img_tensor, mean=0.0, std=1.0):
    img_tensor += torch.randn(img_tensor.size()) * std + mean
    return torch.clip(img_tensor, 0.0, 1.0).type('float')
