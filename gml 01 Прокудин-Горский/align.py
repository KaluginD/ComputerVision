import skimage.io, skimage.transform
import numpy as np


def mse(image1, image2):
    return np.mean((image1 - image2) ** 2)


def cross_entropy(image1, image2):
    return np.sum(image1 * image2) / np.sqrt(np.sum(image1 ** 2) * np.sum(image2 ** 2))


def move(first_picture, second_picture, delta_h, delta_w):
    h, w = first_picture.shape
    if delta_h >= 0 and delta_w >= 0:
        first_new = first_picture[delta_h:, delta_w:]
        second_new = second_picture[:h-delta_h, :w-delta_w]
    elif delta_h < 0 and delta_w >= 0:
        first_new = first_picture[:delta_h, delta_w:]
        second_new = second_picture[-delta_h:, :w-delta_w]
    elif delta_h >= 0 and delta_w < 0:
        first_new = first_picture[delta_h:, :delta_w]
        second_new = second_picture[:h-delta_h, -delta_w:]
    elif delta_h < 0 and delta_w < 0:
        first_new = first_picture[:delta_h, :delta_w]
        second_new = second_picture[-delta_h:, -delta_w:]
    return first_new, second_new


def pyramid_align_pair(first, second):
    if np.max(first.shape) < 400:
        return align_pair(first, second)
    else:
        add = 2 * pyramid_align_pair(
            skimage.transform.rescale(first, 0.5, mode="reflect"),
            skimage.transform.rescale(second, 0.5, mode="reflect")
        )
        _, _, delta_h, delta_w = align_pair(np.roll(np.roll(first, add[2], axis=0), add[3], axis=1), second, 2)
        return add[0], add[1], add[2] + delta_h, delta_w


def align_pair(first, second, step=15):
    mse_best, delta_h_best, delta_w_best = 1e10, 0, 0
    first_best, second_best = first, second

    for delta_h in range(-step, step + 1):
        for delta_w in range(-step, step + 1):
            first_curr, second_curr = move(first, second, delta_h, delta_w)
            mse_curr = mse(first_curr, second_curr)
            if mse_curr < mse_best:
                mse_best = mse_curr
                delta_h_best, delta_w_best = delta_h, delta_w
                first_best, second_best = first_curr, second_curr

    return first_best, second_best, delta_h_best, delta_w_best


def align(img, g_coord):
    W = img.shape[0] // 3
    print(W)
    im_red = img[:W].astype(np.float32) / 255
    im_green = img[W: W * 2].astype(np.float32) / 255
    im_blue = img[W * 2: W * 3].astype(np.float32) / 255

    h, w = im_red.shape
    h_to_cut, w_to_cut = h // 10, w // 10

    im_red = im_red[h_to_cut: -h_to_cut, w_to_cut: -w_to_cut]
    im_green = im_green[h_to_cut: -h_to_cut, w_to_cut: -w_to_cut]
    im_blue = im_blue[h_to_cut: -h_to_cut, w_to_cut: -w_to_cut]

    im_red_best, im_green_r_best, delta_r_h_best, delta_r_w_best = pyramid_align_pair(im_red, im_green)
    im_blue_best, im_blue_r_best, delta_b_h_best, delta_b_w_best = pyramid_align_pair(im_blue, im_green)

    r_coord = [g_coord[0] - W + delta_r_h_best, g_coord[1] + delta_r_w_best]
    b_coord = [g_coord[0] + W + delta_b_h_best, g_coord[1] + delta_b_w_best]
    print(delta_r_h_best, delta_r_w_best, delta_b_h_best, delta_b_w_best)
    return img, r_coord, b_coord

