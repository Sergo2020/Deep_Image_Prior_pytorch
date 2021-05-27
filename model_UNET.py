from basic_layers import *
from torch import nn


class AE_model(nn.Module):
    def __init__(self, device, inp_ch=1, out_ch=1,
                 arch=16, depth=3, activ='leak', concat = None):
        super(AE_model, self).__init__()

        self.activ = activ
        self.device = device
        self.out_ch = out_ch
        self.inp_ch = inp_ch
        self.depth = depth
        self.arch = arch

        if concat is None:
            self.concat = [1]*self.depth
        else:
            self.concat = 2*concat
            self.concat[self.concat == 0] = 1

        self.arch_n = []
        self.enc = []
        self.dec = []
        self.layers = []
        self.skip = []

        self.prep_arch_list()
        self.organize_arch()
        self.prep_params()

    def prep_arch_list(self):
        for dl in range(0, self.depth + 1):
            self.arch_n.append((2 ** (dl - 1)) * self.arch)

        self.arch_n[0] = self.inp_ch

    def organize_arch(self):
        for idx in range(len(self.arch_n) - 1):
            self.enc.append(
                Conv_Block(self.arch_n[idx], self.arch_n[idx + 1], activ=self.activ, pool='down_max'))

        self.layers = [Conv_Block(self.arch_n[-1] , self.arch_n[-1], activ=self.activ, pool='up_bilinear')]

        for idx in range(len(self.arch_n) - 2):
            self.dec.append(
                Conv_Block(self.concat[- (idx + 1)] * self.arch_n[- (idx + 1)], self.arch_n[- (idx + 2)], activ=self.activ, pool='up_bilinear'))
        self.dec.append(Conv_Block(self.concat[0] * self.arch, self.arch, activ=self.activ))
        self.layers.append(Conv_Layer(self.arch, self.out_ch, 1, 1, activ ='sigmoid'))

    def prep_params(self):
        for blk_idx in range(len(self.enc)):
            self.add_module(f'enc_{blk_idx + 1}', self.enc[blk_idx])

        self.add_module(f'mid', self.layers[0])

        for blk_idx in range(len(self.dec)):
            self.add_module(f'dec_{blk_idx + 1}', self.dec[blk_idx])

        self.add_module(f'final', self.layers[1])

    def forward(self, img):
        h = img
        h_skip = []

        for conv in self.enc:
            hs, h = conv(h)
            h_skip.append(hs)

        _, h = self.mid(h)

        for l_idx in range(len(self.dec)):
            if self.concat[-(l_idx + 1)] == 2:
                _, h = self.dec[l_idx](concat_curr(h_skip[-(l_idx + 1)], h))
            else:
                _, h = self.dec[l_idx](h)

        h = self.final(h)

        return h

# ----------------Test---------------------------

'''
if __name__ == '__main__':
    import torch

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    x = torch.randn(5, 1, 100, 100).to(device)

    net = AE_model(device, 1, 1, 16, 4).to(device)
    #y = net(x)

    for name, param in net.named_parameters():
        if param.requires_grad:
            print(name)

    list = net.retriev_params()
    print(list)
'''