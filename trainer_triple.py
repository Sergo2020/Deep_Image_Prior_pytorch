import matplotlib.pyplot as plt
import numpy as np
from torch import optim
from torchvision.utils import save_image

import model_UNET as unet
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

        self.train_bg_loss = []
        self.train_spr_loss = []
        self.train_mask_loss = []

        # Model initialization
        self.extended_l1 = ExtendedL1Loss().to(self.device)
        self.simple_l1 = nn.L1Loss().to(self.device)
        self.gray_loss = GrayLoss().to(self.device)
        self.excl = ExclusionLoss()

        self.bg_net = unet.AE_model(self.device, self.ch_i, self.ch_o, self.arch,
                                    activ='leak', depth=self.depth, concat=self.concat).to(self.device)
        self.spr_net = unet.AE_model(self.device, self.ch_i, self.ch_o, self.arch,
                                     activ='leak', depth=self.depth, concat=self.concat).to(self.device)
        self.mask_net = unet.AE_model(self.device, self.ch_i, 1, self.arch,
                                      activ='leak', depth=self.depth, concat=self.concat).to(self.device)

        self.reset_optimizer()

    def train_step(self, spr_dummy, bg_dummy, msk_dummy, spr_img, spr_hint, bg_hint,
                   it, stage):
        self.clear_grads()

        out_spr = self.spr_net(spr_dummy)
        out_bg = self.bg_net(bg_dummy)
        out_m = self.mask_net(msk_dummy)

        if stage == 'a':

            loss_spr = self.extended_l1(out_spr, spr_img, spr_hint)
            loss_bg = self.extended_l1(out_bg, spr_img, bg_hint)
            loss_m = self.simple_l1(out_m, spr_hint)

            loss = loss_m + loss_spr + loss_bg

        elif stage == 'b':
            loss_spr = self.extended_l1(out_spr, spr_img, out_m)
            loss_bg = self.extended_l1(out_bg, spr_img, 1 - out_m)
            loss_m = 0.5 * self.simple_l1(out_m * out_spr.detach() + (1 - out_m) * out_bg.detach(), spr_img)
            loss_m += 0.01 * (it // 100) * self.gray_loss(out_m)

            loss = loss_m + loss_spr + loss_bg

        loss.backward()
        self.step()

        self.train_spr_loss.append(loss_spr.item())
        self.train_bg_loss.append(loss_bg.item())
        self.train_mask_loss.append(loss_m.item())

        return out_spr.detach(), out_bg.detach(), out_m.detach()

    def integrate_weights(self, model_src, model_dest):
        for p_src, p_dest in zip(model_src.parameters(), model_dest.parameters()):
            p_dest.data = p_src.data.clone().detach()

    def reset_optimizer(self):
        self.optimizer_bg = optim.AdamW(self.bg_net.parameters(), lr=self.init_lr, weight_decay=self.weight_decay)
        self.optimizer_spr = optim.AdamW(self.spr_net.parameters(), lr=self.init_lr, weight_decay=self.weight_decay)
        self.optimizer_msk = optim.AdamW(self.mask_net.parameters(), lr=self.init_lr, weight_decay=self.weight_decay)

    def clear_grads(self):
        self.optimizer_bg.zero_grad()
        self.optimizer_spr.zero_grad()
        self.optimizer_msk.zero_grad()

    def step(self):
        self.optimizer_bg.step()
        self.optimizer_spr.step()
        self.optimizer_msk.step()

    def save_out(self, out_put, title, it):
        save_image(out_put, self.imag_path / f'epoch_{title}_{it}.png')

    def test_save(self, dummy, it):
        self.mask_net.eval()
        out = self.mask_net(dummy)
        self.save_out(out, it)

    def plot_loss(self):
        plt.figure(figsize=(10, 6))
        plt.title('Loss')
        if len(self.train_bg_loss) > 0:
            plt.plot(np.arange(1, len(self.train_bg_loss) + 1), self.train_bg_loss, label='BG', c='g')
        if len(self.train_spr_loss) > 0:
            plt.plot(np.arange(1, len(self.train_spr_loss) + 1), self.train_spr_loss, label='SPR', c='b')
        if len(self.train_mask_loss) > 0:
            plt.plot(np.arange(1, len(self.train_mask_loss) + 1), self.train_mask_loss, label='MASK', c='r')
        plt.grid(True)
        plt.legend(fontsize=12)
        plt.xlabel('Epochs')
        plt.ylabel('Mean Sample Loss')
        plt.show()

    def Save_Models(self, path):
        data_dict = {'BG state': self.bg_net.state_dict(),
                     'SPR state': self.spr_net.state_dict(),
                     'MASK state': self.mask_net.state_dict(),
                     'Train BG Loss': self.train_bg_loss,
                     'Train SPR Loss': self.train_spr_loss,
                     'Train MASK Loss': self.train_mask_loss}
        torch.save(data_dict, path)

    def Load_Model(self, path):
        data_dict = torch.load(path)

        self.mask_net.load_state_dict(data_dict['G state'])
        self.train_bg_loss = data_dict['Train G Loss']

        return len(self.train_bg_loss)
