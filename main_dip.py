from pathlib import Path

from tqdm import tqdm

import data as data
import trainer as tr
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------- Initialization ----------------------------------------
def train(spore_path, iter_num, image_res, lr, factor,
          task=None, mask_path=None, load_path=None):  # task : 'inpaint', 'denoise'
    check_dir = f'check_point/'
    image_dir = Path(image_res)

    check_existence(check_dir, True)
    check_existence(image_dir, True)
    check_existence(spore_path, False)

    # --------------------- Hyperparametrs setup ----------------------------------

    HYPER = {'Epochs': iter_num, 'Noise type': 'uniform',
             'LR': lr, 'Device': device, 'Weights Decay': 0.0,
             'Inp. Channel': 1, 'Out. Channel': 2, 'Arch.': 8, 'Depth': 4,
             'Concat': [1, 1, 1, 1], 'Image folder': image_dir}

    dmy, img = data.prep_io(spore_path, HYPER['Inp. Channel'], HYPER['Out. Channel'],
                            HYPER['Depth'], 'uniform')

    if task == 'inpaint':
        mask = data.prep_inpaint_mask(mask_path, HYPER['Depth'])
        tr.save_image(img[0] * mask[0], image_dir / 'ref_img.png')
        mask = mask.to(device)

    elif task == ' denoise':
        img = data.add_noise(img, std=0.2)
        tr.save_image(img[0], image_dir / 'ref_img.png')

    elif task == 'filter':
        mask = data.prep_inpaint_mask(mask_path, HYPER['Depth'], True)
        img_dft = data.get_dft(img).to(device)
        mask_dft = data.get_dft(mask).to(device)
        mask = mask.to(device)

        tr.save_image(img[0], image_dir / 'ref_img.png')
        tr.save_image(mask[0], image_dir / 'ref_mask.png')
        tr.save_image(data.magnitude(img_dft)[0], image_dir / 'dft_img.png')
        tr.save_image(data.magnitude(mask_dft)[0], image_dir / 'dft_msk.png')
    else:
        tr.save_image(img[0], image_dir / 'ref_img.png')

    dmy, img = dmy.to(device), img.to(device)

    trainer = tr.Trainer(HYPER).to(device)

    trainer.init_train(img, 0.003)

    if not load_path is None:
        trainer.load_model(check_dir + load_path)
    AE_CHK = 0

    epochs = list(range(1 + AE_CHK, iter_num + AE_CHK + 1))
    pbar = tqdm(total=len(epochs), desc='')

    exp_weight = 0.99
    out_avg = None

    for ep in epochs:
        if task == 'inpaint':
            out = trainer.train_inpainter(img, mask)
        elif task == 'filter':
            out = trainer.train_filter(dmy, img, img_dft, mask, mask_dft)
        else:
            out = trainer.train_denoiser(dmy)

        if out_avg is None:
            out_avg = out
        else:
            out_avg = out_avg * exp_weight + out * (1 - exp_weight)

        pbar.update()

        if (ep % factor) == 0:
            if ep > 0:
                trainer.plot_loss(ep)

                if task == 'filter':
                    trainer.save_out(out, ep)
                else:
                    trainer.save_out(out, ep)
                # trainer.Save_Model(check_dir + '/chk_' + str(ep) + '.pt')

        pbar.postfix = f'Loss {trainer.train_g_loss[- 1]:.5f}'

    trainer.save_model(check_dir + '/chk_last.pt', True)
    torch.cuda.empty_cache()


def test_model(load_path, spore_path):
    check_existence(spore_path, False)
    check_existence(load_path, False)

    trainer = torch.load(load_path)

    dmy, img = data.prep_io(spore_path, trainer.ch_i, trainer.ch_o,
                            trainer.depth, 'special')

    # dmy, img = dmy.to(device), img.to(device)
    img, dmy = dmy.to(device), img.to(device)

    dmy_saved = dmy.clone()
    dmy = dmy_saved + torch.randn_like(dmy_saved) * 0.003

    out = trainer.test_model(dmy)

    out = pil2np(out[0].cpu())
    return out


if __name__ == '__main__':
    spore_path = Path(r'D:\SporeData\Test_Images\dft_test\07_09_20_1_CA_spore.png')
    bg_path = Path(r'D:\SporeData\Test_Images\dft_test\07_09_20_1_CA_bg.png')
    label_path = Path(r'D:\SporeData\Test_Images\dft_test\07_09_20_1_CA_spore_lbl.png')

    save_image_dir = Path('Results')

    img_out = train(spore_path, 500, save_image_dir, 1e-3, 100,
                    mask_path=label_path, task='filter')

    print('Done')
