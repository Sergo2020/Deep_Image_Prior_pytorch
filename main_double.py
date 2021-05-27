from pathlib import Path

import torch
from tqdm import tqdm

import data as data
import trainer_triple as tr
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------- Initialization ----------------------------------------
def train(spore_path, bg_path, iter_a, iter_b, image_res, lr, factor):
    check_dir = f'check_point_spores/'
    image_dir = Path(image_res)

    check_existence(check_dir, True)
    check_existence(image_dir, True)
    check_existence(spore_path, False)
    check_existence(bg_path, False)

    # --------------------- Hyperparametrs setup ----------------------------------

    HYPER = {'Noise type': 'uniform',
             'LR': lr, 'Device': device, 'Weights Decay': 0.0,
             'Inp. Channel': 1, 'Out. Channel': 1, 'Arch.': 8, 'Depth': 3,
             'Concat': [0, 0, 0], 'Image folder': image_dir}

    spr_dmy, spr_image, spr_slcy = data.prep_io(spore_path,
                                                HYPER['Inp. Channel'],HYPER['Depth'],
                                                'uniform', saliecity=True)

    bg_dmy, _ = data.prep_io(bg_path, HYPER['Inp. Channel'],
                                             HYPER['Depth'], 'uniform')

    msk_dmy = data.init_dummy('uniform', bg_dmy.size()[2:], HYPER['Inp. Channel'])

    spr_slcy[spr_slcy > 0.2] = 1.0
    spr_slcy *=0.5

    tr.save_image(spr_image[0], image_dir / 'ref_spore.png')
    tr.save_image(spr_slcy[0], image_dir / 'ref_saliecity.png')
    #tr.save_image(torch.abs(bg_image - spr_image)[0], image_dir / 'ref_sub.png')

    bg_dmy, spr_dmy, msk_dmy = bg_dmy.to(device), spr_dmy.to(device), msk_dmy.to(device)
    spr_image = spr_image.to(device)
    spr_slcy, bg_slcy = spr_slcy.to(device).detach(), (1 - spr_slcy).to(device).detach()

    trainer = tr.Trainer(HYPER)

    epochs = list(range(1, iter_a + iter_b + 1))
    pbar = tqdm(total=iter_a + iter_b, desc='')

    bg_dmy_saved = bg_dmy.detach().clone()
    spr_dmy_saved = spr_dmy.detach().clone()
    msk_dmy_saved = msk_dmy.detach().clone()

    stage = 'a'
    for ep in epochs:
        bg_dmy = bg_dmy_saved + torch.randn_like(bg_dmy_saved) * 0.003
        spr_dmy = spr_dmy_saved + torch.randn_like(spr_dmy_saved) * 0.003
        msk_dmy = msk_dmy_saved + torch.randn_like(msk_dmy_saved) * 0.003

        if ep == iter_a: stage = 'b'

        out_spr, out_bg, out_m = trainer.train_step(spr_dmy, bg_dmy, msk_dmy,
                                                    spr_image, spr_slcy, bg_slcy,
                                                    ep, stage = stage)

        pbar.update()

        if (ep % factor) == 0:
            if ep > 0:
                trainer.plot_loss()
                trainer.save_out(out_bg, 'bg', ep)
                trainer.save_out(out_spr, 'spr', ep)
                trainer.save_out(out_m, 'mask', ep)

        pbar.postfix = f'Stage {stage} : BG Loss {trainer.train_bg_loss[- 1]:.5f}' + \
                       f'| SPR Loss {trainer.train_spr_loss[- 1]:.5f}' + \
                       f'| MSK Loss {trainer.train_mask_loss[- 1]:.5f}'

    trainer.Save_Models(check_dir + '/chk_last.pt')

    torch.cuda.empty_cache()


if __name__ == '__main__':

    spore_path = Path(r'D:\SporeData\Test_Images\dip_test\chopper.jpg')
    bg_path = Path(r'D:\SporeData\Test_Images\dip_test\chopper.jpg')
    i_a = 1000
    i_b = 2000

    for lr in [1e-3]:
        train(spore_path, bg_path, i_a, i_b, f'Results_o', lr, factor=int(i_a / 2))
