import os

import torch
import cv2 as cv
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


def check_existence(path, create=False):
    if not os.path.exists(path):
        print("Creating check point directory - " + str(path))

        if create:
            os.mkdir(path)
        else:
            print(f'{str(path)}\nPath not found')
            exit()

def crop_from_points(img_dim, crop_size, overlap=0):  # Overlap in ratio to crop size
    points = [0]
    stride = int(crop_size * (1 - overlap))
    counter = 1
    while True:
        pt = stride * counter
        if pt + crop_size >= img_dim:
            points.append(img_dim - crop_size)
            break
        else:
            points.append(pt)
        counter += 1
    return points


def prep_crop_coord(img_shape, crop_size, overlap):
    Ys = crop_from_points(img_shape[0], crop_size, overlap)
    Xs = crop_from_points(img_shape[1], crop_size, overlap)

    return Ys, Xs


def image_cropper(img, ys, xs, crop_size):
    crop_dict = {}

    cnt = 0
    for y in ys:
        crop_dict[y] = {}
        for x in xs:
            crop = img[y: y + crop_size, x: x + crop_size]
            crop_dict[y][x] = crop
            cnt += 1

    return crop_dict


def crop_Imager(crop_dict, ys, xs, ovr, img_shape):
    n_img = np.zeros((img_shape))
    crop_size = crop_dict[0][0].shape[0]
    ovr = int(crop_size * ovr)

    for idx_y in range(len(ys)):
        for idx_x in range(len(xs)):
            y = ys[idx_y]
            x = xs[idx_x]

            if idx_x > 0:
                xp = xs[idx_x - 1]
                n_img[y: y + crop_size, x: x + ovr] = (crop_dict[y][xp][:, crop_size - ovr:] + crop_dict[y][x][:,
                                                                                               :ovr]) / 2
                n_img[y: y + crop_size, x + ovr: x + crop_size] = crop_dict[y][x][:, ovr:]

            elif idx_y > 0:
                yp = ys[idx_y - 1]
                n_img[y: y + ovr, x: x + crop_size] = (crop_dict[yp][x][crop_size - ovr:, :] + crop_dict[y][x][
                                                                                               :ovr, ]) / 2
                n_img[y + ovr: y + crop_size, x: x + crop_size] = crop_dict[y][x][ovr:, :]

            elif y > 0 and x > 0:
                xp = xs[idx_x - 1]
                yp = ys[idx_y - 1]
                n_img[y: y + ovr, x: x + ovr] = (crop_dict[yp][xp][crop_size - ovr:, crop_size - ovr:] + crop_dict[y][
                                                                                                             x][:ovr,
                                                                                                         :ovr]) / 2
                n_img[y + ovr: y + crop_size, x + ovr: x + crop_size] = crop_dict[y][x][ovr:, ovr:]

            else:
                n_img[y: y + crop_size, x: x + crop_size] = crop_dict[y][x]

    return n_img


def remove_frame(img, inside=False):
    _, n_img = cv.threshold(img.copy(), thresh=200, maxval=255, type=cv.THRESH_BINARY_INV)
    idxes = np.where(n_img > 0)
    upper_left = min(idxes[0]), min(idxes[1])
    bottom_right = max(idxes[0]), max(idxes[1])

    img = img[upper_left[0]: bottom_right[0], upper_left[1]: bottom_right[1]]

    _, n_img = cv.threshold(img.copy(), thresh=200, maxval=255, type=cv.THRESH_BINARY_INV)
    idxes = np.where(n_img < 255)
    upper_left = min(idxes[0]), min(idxes[1])
    bottom_right = max(idxes[0]), max(idxes[1])

    img = img[upper_left[0]: bottom_right[0], upper_left[1]: bottom_right[1]]

    return img


def show_image(img, counters=None, centers=None, rectangles=None, title='Image',
               path=None):  # Simple function that shows image in pre set image size without axis and grid
    plt.figure(figsize=(1608 / 96, 1608 / 96), frameon=False)
    img_t = img.copy()

    if centers is not None:
        for idx in range(1, len(centers) + 1):
            cv.putText(img_t, str(idx), tuple(centers[idx - 1]), cv.FONT_HERSHEY_SIMPLEX, 0.2, (255, 0, 0), 3,
                       cv.LINE_AA)

    if rectangles is not None:
        for r in rectangles:
            cv.rectangle(img_t, r[0], r[1], color=(255, 0, 0), thickness=3)

    if counters is not None:
        for cnt in counters:
            cv.drawContours(img_t, cnt, -1, (255, 0, 0), thickness=3)

    if len(img.shape) < 3:
        plt.imshow(img_t, 'gray')
    else:
        plt.imshow(img_t)
    plt.grid(False)
    plt.axis(False)
    plt.title(title)

    if path:
        plt.savefig(path)

    plt.show()


def show_pairs(img_1, img_2, title_1, title_2):
    plt.subplot(1, 2, 1)
    plt.imshow(img_1, cmap='gray')
    plt.title(title_1)
    plt.grid(False)
    plt.axis(False)

    plt.subplot(1, 2, 2)
    plt.imshow(img_2, cmap='gray')
    plt.title(title_2)
    plt.grid(False)
    plt.axis(False)
    plt.show()


def open_image(path, tp='cv_gray'):
    if tp == 'cv_rgb':
        img = cv.imread(str(path))
        return cv.cvtColor(img, cv.COLOR_BGR2RGB)
    elif tp == 'cv_gray':
        return cv.imread(str(path), 0)
    elif tp == 'pil_rgb':
        return Image.open(path).convert('RGB')
    elif tp == 'pil_gray':
        return Image.open(path).convert('L')


def pil2np(img):
    arr = np.array(img)
    if arr.shape[0] == 3:
        return np.transpose(arr, (1, 2, 0))
    elif arr.shape[0] == 1:
        return arr.squeeze(0)
    else:
        return arr


def np2pil(arr):
    return Image.fromarray(arr)
