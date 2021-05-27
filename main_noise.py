import os
from pathlib import Path

import data as data
import trainer as tr
from tqdm import tqdm
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------- Initialization ----------------------------------------
def train_model(bg_path_load, bg_path, object_path, image_save_path,
                iter_num, factor):

    check_existence(bg_path, False)
    check_existence(bg_path_load, False)
    check_existence(bg_path, False)
    check_existence(object_path, False)
    check_existence(image_save_path, True)

    trainer_bg = torch.load(bg_path_load) # Prev model and noise
    trainer_bg.freeze_net()

    HYPER = {'Epochs': iter_num, 'Noise type': 'uniform',
             'LR': lr, 'Device': device, 'Weights Decay': 0.0,
             'Inp. Channel': trainer_bg.ch_i, 'Out. Channel': trainer_bg.ch_i, 'Arch.': 8, 'Depth': 4,
             'Concat': [0, 0, 1, 1], 'Image folder': image_save_path}

    _, img = data.prep_io(object_path, trainer_bg.ch_i, trainer_bg.ch_o,
                            trainer_bg.depth, 'mesh')
    dmy = trainer_bg.noise
    dmy, img = dmy.to(device), img.to(device)

    trainer_noise = tr.Trainer(HYPER).to(device)

    epochs = list(range(1, iter_num + 1))
    pbar = tqdm(total=len(epochs), desc='')

    for ep in epochs:
        out = trainer_noise.train_noise(dmy, trainer_bg.G, img)

        pbar.update()

        if (ep % factor) == 0:
            if ep > 0:
                trainer_noise.plot_loss(ep)
                trainer_noise.save_out(out, ep)
                #trainer.Save_Model(check_dir + '/chk_' + str(ep) + '.pt')

        pbar.postfix = f'Loss {trainer_noise.train_g_loss[- 1]:.5f}'

def test_model(load_path, spore_path, save_path):

    check_existence(spore_path, False)
    check_existence(load_path, False)

    trainer = torch.load(load_path)

    dmy = trainer.prep_noise()

    out = trainer.test_model(dmy)
    tr.save_image(out,  save_path / 'test.png')

    return out.cpu()

if __name__ == '__main__':

    obj_path = Path(r'D:\SporeData\Test_Images\dip_test\snail.jpg')
    bg_path = Path(r'D:\SporeData\Test_Images\dip_test\snail.jpg')
    save_path = Path(f'Results_o')
    load_path = Path('check_point/chk_last.pt')

    out = test_model(load_path, obj_path, save_path)

    for lr in [1e-4]:
        img_out = train_model(load_path, bg_path, obj_path, save_path, 2000, 500)

    print('Done')

