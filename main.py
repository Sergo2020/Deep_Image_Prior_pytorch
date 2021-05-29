from pathlib import Path

from tqdm import tqdm

import data as data
import trainer as tr
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------- Initialization ----------------------------------------
def train(img_path, image_res, hyper_pars,
          task=None, mask_path=None, load_path=None):

    check_dir = f'check_point/'

    check_existence(check_dir, True)
    check_existence(image_res, True)
    check_existence(img_path, False)

    # --------------------- Hyperparametrs setup ----------------------------------

    dmy, img = data.prep_io(img_path, hyper_pars['Inp. Channel'],
                            hyper_pars['Depth'], 'uniform')

    if task == 'inpaint':
        mask = data.prep_inpaint_mask(mask_path, hyper_pars['Depth'])
        tr.save_image(img[0] * mask[0], image_res / 'ref_img.png')
        mask = mask.to(device)

    elif task == 'denoise':
        img = data.add_noise(img, std=0.2)
        tr.save_image(img[0], image_res / 'ref_img.png')
    elif task == 'jpeg':
        tr.save_image(img[0], image_res / 'ref_img.png')


    dmy, img = dmy.to(device), img.to(device)

    trainer = tr.Trainer(hyper_pars).to(device)

    trainer.init_train(dmy, 0.003)

    if not load_path is None:
        trainer.load_model(check_dir + load_path)
    AE_CHK = 0

    epochs = list(range(1 + AE_CHK, hyper_pars['Epochs'] + AE_CHK + 1))
    pbar = tqdm(total=len(epochs), desc='')

    exp_weight = 0.99
    out_avg = None

    for ep in epochs:
        if task == 'inpaint':
            out = trainer.train_inpainter(img, mask)
        elif task == 'denoise' or task == 'jpeg':
            out = trainer.train_denoiser(img)
        else:
            print('Wrong task selected. Two optionas are availabel :\n inpaint\n denoise\n jpeg ')

        if out_avg is None:
            out_avg = out
        else:
            out_avg = out_avg * exp_weight + out * (1 - exp_weight)

        pbar.update()

        if (ep % hyper['Factor']) == 0:
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


if __name__ == '__main__':

    root_dir = get_curr_path()

    image_path = Path(get_curr_path() + r'\images\car.png')
    mask_path = Path(get_curr_path() + r'\images\car_mask.png')

    result_path = Path(get_curr_path() + r'\results')

    hyper = {'Epochs': 1000, 'Factor' : 100, 'Noise type': 'uniform',
             'LR': 1e-3, 'Device': device, 'Weights Decay': 0.0,
             'Inp. Channel': 16, 'Out. Channel': 3, 'Arch.': 16, 'Depth': 4,
             'Concat': [1, 0, 0, 1], 'Image folder': result_path}

    img_out = train(image_path, result_path, hyper,
                    mask_path=mask_path, task='inpaint')

    print('Done')