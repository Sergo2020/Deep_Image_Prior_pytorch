import matplotlib.pyplot as plt
from torchvision.utils import save_image
import numpy as np
from torch import optim

import model_UNET as unet

import utils as u
from loss import *


class Trainer(nn.Module):
    def __init__(self, hyperparams):
        super(Trainer, self).__init__()

        # Hyperparameters
        self.device = hyperparams['Device']
        self.init_lr = hyperparams['LR']
        self.imag_path = hyperparams['Image folder']
        self.weight_decay = hyperparams['Weights Decay']
        self.ch_i = hyperparams['Inp. Channel']
        self.ch_o = hyperparams['Out. Channel']
        self.arch = hyperparams['Arch.']
        self.depth = hyperparams['Depth']
        self.concat = np.array(hyperparams['Concat'])
        self.softmax = nn.Softmax(dim = 1)

        self.train_g_loss = []

        self.noise = None

        # Model initialization
        self.loss = nn.MSELoss(reduction  = 'mean').to(self.device)
        self.G = unet.AE_model(self.device, self.ch_i, self.ch_o, self.arch,
                               activ = 'leak', depth=self.depth, concat=self.concat).to(self.device)
        self.optimizer_G = optim.AdamW(self.G.parameters(),
                                       lr=self.init_lr, weight_decay=self.weight_decay)

    def init_train(self, noise, var):
        self.noise = noise
        self.var = var

    def prep_noise(self, var = -1):
        if var == -1:
            return self.noise + torch.randn_like(self.noise.detach()) * self.var
        else:
            return self.noise + torch.randn_like(self.noise.detach()) * var

    def freeze_net(self):
        for p in self.G.parameters():
            p.requires_grad = False

    def train_denoiser(self, corrupt_img):
        self.G.train()

        self.optimizer_G.zero_grad()

        dmy = self.prep_noise()

        out = self.G(dmy)
        loss = self.loss(out, corrupt_img)

        loss.backward()
        self.optimizer_G.step()

        self.train_g_loss.append(loss.item())

        return out.detach()

    def train_inpainter(self, corrupt_img, mask):
        self.G.train()

        self.optimizer_G.zero_grad()

        dmy = self.prep_noise()

        out = self.G(dmy)
        loss = self.loss(out*mask, corrupt_img*mask)

        loss.backward()
        self.optimizer_G.step()

        self.train_g_loss.append(loss.item())

        return out.detach()

    def train_filter(self, dmy, img, dft_img, label, dft_label):
        self.G.train()

        self.optimizer_G.zero_grad()

        dmy = self.prep_noise()

        out = self.G(img)

        #dft_l = out*dft_img

        loss = self.loss(out, dft_img)

        loss.backward()
        self.optimizer_G.step()

        self.train_g_loss.append(loss.item())

        return out.detach()

    def test_model(self, dmy):
        #self.G.train()

        with torch.no_grad():
            out = self.G(dmy)
            self.save_out(out, 'test')

        return out

    def train_noise(self, inp_noise, bg_model, obj_img):
        self.G.train()

        self.optimizer_G.zero_grad()

        out_noise = self.G(inp_noise)
        out_img = bg_model(out_noise)

        loss = self.loss(out_img, obj_img)

        loss.backward()
        self.optimizer_G.step()

        self.train_g_loss.append(loss.item())

        return out_img.detach()

    def save_out(self, out_put, it):
        save_image(out_put, self.imag_path / f'epoch_{it}.png')

    def test_save(self, dummy, it):
        self.G.eval()
        out = self.G(dummy)
        self.save_out(out, it)


    def img2spore(self, img, th=0.7, sigmoid=False, softmax=False):
        self.AE.eval()
        img = img.to(self.device)
        with torch.no_grad():
            spr_map = self.AE(img).cpu()
            torch.cuda.empty_cache()
            if sigmoid:
                spr_map = torch.sigmoid(spr_map)
            if softmax:
                spr_map = softmax(spr_map)

        spr_map[spr_map < th] = 0
        _, ch, h, w = spr_map.size()

        if ch > 1:
            spr_map = u.pil2np(spr_map[0])
        else:
            spr_map = spr_map.view(h, w).numpy()
        return spr_map

    def plot_loss(self, eps):
        if eps == 0:
            eps = 1
        plt.figure(figsize=(10, 6))
        plt.title('Loss')
        plt.plot(np.arange(1, len(self.train_g_loss) + 1), self.train_g_loss, label='MSE', c='g')
        plt.grid(True)
        plt.legend(fontsize=12)
        plt.xlabel('Epochs')
        plt.ylabel('Mean Sample Loss')
        plt.show()

    def save_model(self, path, full_model = False):

        if not full_model:
            data_dict = {'G state': self.G.state_dict(),
                         'Train G Loss': self.train_g_loss}
            torch.save(data_dict, path)
        else:
            torch.save(self, path)

    def load_model(self, path):
        data_dict = torch.load(path)

        self.G.load_state_dict(data_dict['G state'])
        self.train_g_loss = data_dict['Train G Loss']

        return len(self.train_g_loss)



