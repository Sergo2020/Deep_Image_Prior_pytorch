import matplotlib.pyplot as plt
from torchvision.utils import save_image
import numpy as np
from torch import optim

import model

import utils as u
from loss import *


class Trainer(nn.Module):
    def __init__(self, hyperparams):
        super(Trainer, self).__init__()

        # Hyperparameters
        self.device = hyperparams['Device']
        self.init_lr = hyperparams['LR']
        self.ch_i = hyperparams['Inp. Channel']
        self.ch_o = hyperparams['Out. Channel']
        self.arch = hyperparams['Arch.']
        self.depth = hyperparams['Depth']
        self.concat = np.array(hyperparams['Concat'])

        self.train_g_loss = []
        self.noise = None

        # Model initialization
        self.loss = nn.MSELoss(reduction='mean').to(self.device)
        self.AE = model.Unet(self.device, self.ch_i, self.ch_o, self.arch,
                             activ='leak', depth=self.depth, concat=self.concat).to(self.device)
        self.optimizer = optim.AdamW(self.AE.parameters(), lr=self.init_lr)

    def init_train(self, noise, var):
        self.noise = noise
        self.var = var

    def prep_noise(self, var=-1):
        if var == -1:
            return self.noise + torch.randn_like(self.noise.detach()) * self.var
        else:
            return self.noise + torch.randn_like(self.noise.detach()) * var

    def train_denoiser(self, corrupt_img):
        self.AE.train()

        self.optimizer.zero_grad()

        dmy = self.prep_noise()

        out = self.AE(dmy)
        loss = self.loss(out, corrupt_img)

        loss.backward()
        self.optimizer.step()

        self.train_g_loss.append(loss.item())

        return out.detach().cpu()

    def train_inpainter(self, corrupt_img, mask):
        self.AE.train()

        self.optimizer.zero_grad()

        dmy = self.prep_noise()

        out = self.AE(dmy)
        loss = self.loss(out * mask, corrupt_img * mask)

        loss.backward()
        self.optimizer.step()

        self.train_g_loss.append(loss.item())

        return out.detach().cpu()

    @staticmethod
    def save_out(out_put, dir, fl_name):
        save_image(out_put, dir / (fl_name + ".png"))

    def test_save(self, dummy, it):
        self.AE.eval()
        out = self.AE(dummy)
        self.save_out(out, it)

    def plot_loss(self):
        plt.figure(figsize=(10, 6))
        plt.title('Loss')
        plt.plot(np.arange(1, len(self.train_g_loss) + 1), self.train_g_loss, label='MSE', c='g')
        plt.grid(True)
        plt.legend(fontsize=12)
        plt.xlabel('Epochs')
        plt.ylabel('Mean Sample Loss')
        plt.show()

    def save_model(self, path, full_model=False):

        if not full_model:
            data_dict = {'G state': self.AE.state_dict(),
                         'Train G Loss': self.train_g_loss}
            torch.save(data_dict, path)
        else:
            torch.save(self, path)

    def load_model(self, path):
        data_dict = torch.load(path)

        self.AE.load_state_dict(data_dict['G state'])
        self.train_g_loss = data_dict['Train G Loss']

        return len(self.train_g_loss)
