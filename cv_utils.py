from scipy.spatial import cKDTree

from utils import *


# --- Algorithms ---

def sift_homography(src_img, dst_img, spore_crds=None, min_match=20):
    """
    Image to image mapping using SIFT and homography.
    Output is: visual location of tranlated image, translated image, rectangle points
    """

    sift = cv.xfeatures2d.SIFT_create()

    # Find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(src_img, None)
    kp2, des2 = sift.detectAndCompute(dst_img, None)
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    # If enough points find Homography matrix and transform image
    if len(good) > 20:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        m, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        src_translated = cv.warpPerspective(src_img, m, (dst_img.shape[0], dst_img.shape[1]))
        h, w = src_img.shape[0], src_img.shape[1]
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv.perspectiveTransform(pts, m)

        if spore_crds is not None:
            spore_crds = np.expand_dims(spore_crds, 1).astype(float)
            spore_crds = cv.perspectiveTransform(spore_crds, m)
            spore_crds = spore_crds.squeeze(1).astype(int)
        if len(dst_img.shape) < 3:
            dst_img_clr = cv.cvtColor(dst_img, cv.COLOR_GRAY2RGB)
            dst_boundry = cv.polylines(np.zeros_like(dst_img_clr), [np.int32(dst)], True, (0, 0, 255), 3, cv.LINE_AA)
        else:
            dst_boundry = cv.polylines(np.zeros_like(dst_img), [np.int32(dst)], True, (0, 0, 255), 3, cv.LINE_AA)

    else:
        print("Not enough matches are found - {}/{}".format(len(good), min_match))

        return None, None, None

    return dst_boundry, src_translated, spore_crds


def blob_detector(img):  # input is uint8 image

    c = []
    s = []
    r = []

    params = cv.SimpleBlobDetector_Params()

    params.minThreshold = 0
    params.maxThreshold = 255

    params.filterByCircularity = True
    params.minCircularity = 0.0  # 0.5

    params.filterByInertia = True
    params.minInertiaRatio = 0.0  # .4
    params.maxInertiaRatio = 1.0

    detector = cv.SimpleBlobDetector_create(params)

    keypoints = detector.detect(img)
    # _, th = cv.threshold(img, thresh = 10, maxval = 255, type = cv.THRESH_BINARY_INV)

    for k in keypoints:
        c.append([int(k.pt[0]), int(k.pt[1])])
        s.append(k.size ** 2)
        r.append([(int(c[-1][0] - k.size / 2), int(c[-1][1] - k.size / 2)),
                  (int(c[-1][0] + k.size / 2), int(c[-1][1] + k.size / 2))])

    c = np.array(c)
    r = np.array(r)
    s = np.array(s)

    try:
        idxes = filter_points_kd(c)
        c = c[idxes]
        s = s[idxes]
        r = r[idxes]
    except:
        return 0, 0, 0

    idxes = s < 800
    c = c[idxes]
    s = s[idxes]
    r = r[idxes]

    return c, s, r


def counter_detector(fcn_map, img, show=None, save=None):
    n_map = np.array(255 * fcn_map).astype('uint8')
    n_map[n_map > 25] = 255
    _, map_bin = cv.threshold(n_map, thresh=25, maxval=255, type=cv.THRESH_BINARY_INV)

    cnts, _ = cv.findContours(map_bin, 1, cv.CHAIN_APPROX_NONE)

    n_cnts = []
    centers = []
    rects = []
    areas = []

    for c in cnts:
        a = cv.contourArea(c)
        if a > 5000 or a < 50: continue
        n_cnts.append(c)
        m = cv.moments(c)
        cx = int(m["m10"] / m["m00"])
        cy = int(m["m01"] / m["m00"])
        areas.append(a)
        sd = 2 * int(np.sqrt(a / np.pi))
        centers.append((cx, cy))

        rects.append([(int(cx - sd / 2), int(cy - sd / 2)),
                      (int(cx + sd / 2), int(cy + sd / 2))])
    im_sh = img.copy()
    if show:
        if show == 'map':
            show_image(map_bin, path=save)
        elif show == 'boxes':
            for r in rects:
                cv.rectangle(im_sh, r[0], r[1], color=(0, 255, 0), thickness=3)
            show_image(im_sh, path=save)

        elif show == 'fill':
            show_image(im_sh, counters=n_cnts, path=save)

    return centers, rects, areas, n_cnts


def saliency_detection(img, th=0.4, ker=3):
    saliency_obj = cv.saliency.StaticSaliencyFineGrained_create()
    success, saliency_map = saliency_obj.computeSaliency(img)

    if not success:
        exit()

    if th > 0:
        # saliency_map[saliency_map > th] = 1.0
        saliency_map[saliency_map <= th] = 0.0
    if ker > 0:
        saliency_map = morph(saliency_map, ker, 'dilate')
    return saliency_map


def hist_clahe(img):
    clahe_obj = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe_obj.apply(img)
    return cl1


# --- Supplementary ---

def gauss_blur(img, k_size):
    n_i = cv.GaussianBlur(img, (k_size, k_size), 0)
    return n_i


def blur(img, k_size):
    n_i = cv.blur(img, ksize=(k_size, k_size))
    return n_i


def edges(img):
    return cv.Canny(img, 100, 200)


def morph(img, ker, op):
    kernel = np.ones((ker, ker), dtype='uint8')
    if op == 'close':
        img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
    elif op == 'open':
        img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
    elif op == 'erod':
        img = cv.erode(img, kernel, iterations=2)
    elif op == 'dilate':
        img = cv.dilate(img, kernel, iterations=1)
    return img


def contrast_brigthness(img, alpha, beta):
    img = alpha * img + beta
    img[img > 255] = 255
    return img.astype('uint8')


def auto_adjust(img):
    img = img.astype(float)
    img_range = np.max(img) - np.min(img)
    alpha = 255 / img_range
    beta = -np.min(img) * alpha

    return contrast_brigthness(img, alpha, beta)


def fill_counters(fcn_map, counters):
    fcn = 255 * fcn_map.astype(int)
    z_map = np.zeros_like(fcn_map)
    cv.drawContours(z_map, counters, -1, 255, thickness=-1)

    fcn += z_map.astype(int)
    fcn[fcn > 255] = 255

    return fcn.astype(float) / 255


def filter_points_kd(arr, dist=5):
    mynumbers = [tuple(point) for point in arr]
    tree = cKDTree(mynumbers)  # build k-dimensional trie
    pairs = tree.query_pairs(dist)  # find all pairs closer than radius: r
    neighbors = {}  # dictionary of neighbors
    for i, j in pairs:
        if i not in neighbors.keys():
            neighbors[i] = [j]
        else:
            neighbors[i].append(j)
        if j not in neighbors.keys():
            neighbors[j] = [i]
        else:
            neighbors[j].append(j)

    keep = []
    discard = []  # a list would work, but I use a set for fast member testing with `in`
    for pnt_idx in range(len(mynumbers)):
        if pnt_idx not in discard:  # if node already in discard set: skip
            keep.append(pnt_idx)  # add node to keep list
            try:
                discard.extend([i for i in neighbors[pnt_idx]])
            except:
                continue

    return keep


def img2dft(img, magnitude = True):
    if len(img.shape) == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    dft = cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT)
    dft = np.fft.fftshift(dft)

    if magnitude:
        magnitude_spectrum = dft_magnitute(dft)
        return dft, magnitude_spectrum
    else:
        return dft


def dft_magnitute(ft):
    return 20 * np.log(cv.magnitude(ft[:, :, 0], ft[:, :, 1]))


def dft2img(f_image):
    f_image = np.fft.ifftshift(f_image)
    img_back = cv.idft(f_image)
    img_back = cv.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    return img_back


# ----- Test ----

if __name__ == '__main__':
    from pathlib import Path
    from torchvision.utils import save_image

    path_bg = Path(r'D:\SporeData\Test_Images\dft_test\07_09_20_1_CA_bg.png')
    path_spore = Path(r'D:\SporeData\Test_Images\dft_test\07_09_20_1_CA_spore.png')
    path_lbl = Path(r'D:\SporeData\Test_Images\dft_test\07_09_20_1_CA_spore_lbl.png')

    img_bg = cv.imread(str(path_bg), 0)
    img_spr = cv.imread(str(path_spore), 0)
    img_lbl = cv.imread(str(path_lbl), 0)

    img_mult = img_spr * (img_lbl / 255)
    # _, img_spr, _ = sift_homography(img_spr, img_bg)

    f_bg, m_bg = img2dft(img_bg)
    b_bg = dft2img(f_bg)

    f_s, m_s = img2dft(img_spr)
    b_s = dft2img(f_s)
    ref_img_s = dft2img(f_s/f_s.max())

    f_lbl, m_lbl = img2dft(img_lbl)
    f_lbl = f_lbl / (f_lbl.max() - f_lbl.min())
    b_lbl = dft2img(f_lbl)

    f_mult, m_mult = img2dft(img_mult)
    b_mult = dft2img(f_mult)

    filtered = f_s * f_lbl
    m_1 = dft_magnitute(filtered)
    filtered_img = dft2img(filtered)


    show_pairs(img_bg, m_bg, 'bg original', 'bg dft')
    show_pairs(img_spr, ref_img_s, 'spr original', 'spr idft norm')
    show_pairs(b_lbl, m_lbl, 'lbl original', 'lbl dft')

    #show_pairs(filtered_bs_img / filtered_bs_img.max(), m_bs, 'bs original', 'bs dft')

    print('done')
