import numpy as np


def get_intensity(img):
    weights = np.array([0.299, 0.587, 0.114])
    return img @ weights


def get_intensity_with_mask(intensities, mask):
    intensity_with_mask = intensities.copy()

    h, w = intensities.shape
    intensity_with_mask += h * w * 256 * mask

    return intensity_with_mask


def get_intensities_sum(grad_intensity):
    intensity_sum = grad_intensity.copy()
    h, w = grad_intensity.shape
    for i in range(1, h):
        add = np.zeros(w)
        last_block = np.array([intensity_sum[i - 1, 1:-1], intensity_sum[i - 1, 2:], intensity_sum[i - 1, :-2]])
        add[1:-1] = np.min(last_block,  axis=0)
        add[0] = np.min(intensity_sum[i - 1, :2])
        add[-1] = np.min(intensity_sum[i - 1, -2:])
        intensity_sum[i] += add

    return intensity_sum


def get_min_seam(sum_intensity):
    seam = np.zeros(sum_intensity.shape)
    h, w = sum_intensity.shape[:2]
    y = np.argmin(sum_intensity[h - 1])
    seam[h - 1, y] = 1
    for x in range(h - 1, -1, -1):
        start = max(0, y - 1)
        end = min(y + 2, w)
        diff = np.argmin(sum_intensity[x, start: end])
        y = start + diff
        seam[x, y] = 1
    return seam


def get_grad_intensity(image):
    I_x = np.roll(image, 1, axis=0) - np.roll(image, -1, axis=0)
    I_y = np.roll(image, 1, axis=1) - np.roll(image, -1, axis=1)
    I_x[0, :] = image[1, :] - image[0, :]
    I_x[-1, :] = image[-1, :] - image[-2, :]
    I_y[:, 0] = image[:, 1] - image[:, 0]
    I_y[:, -1] = image[:, -1] - image[:, -2]
    intensity_grad = (I_x ** 2 + I_y ** 2) ** 0.5
    return intensity_grad


def get_seam_mask(seam, shape):
    mask = np.zeros(shape)
    rows = list(range(len(seam)))
    mask[rows, seam] = 1
    return mask


def shrink_seam(new_image, mask, min_seam):
    for index, pixel_to_shrink in enumerate(min_seam):
        new_image[index, pixel_to_shrink: -1] = new_image[index, pixel_to_shrink + 1:]
        if mask is not None:
            mask[index, pixel_to_shrink: -1] = mask[index, pixel_to_shrink + 1:]
    new_image = new_image[:-1]
    if mask is not None:
        mask = mask[:-1]
    return new_image, mask


def expand_seam(new_image, mask, min_seam):
    h, w, d = new_image.shape
    big_image = np.zeros((h, w + 1, d))
    big_image[:, :-1] = new_image
    big_mask = mask
    # print('image shape: ', new_image.shape, 'big image shape: ', big_image.shape)
    if mask is not None:
        big_mask = np.zeros((h, w + 1))
        big_mask[:, :-1] = mask
    for index, pixel_to_expand in enumerate(min_seam):
        if pixel_to_expand == w - 1:
            value_to_add = new_image[index, pixel_to_expand]
        else:
            value_to_add = np.mean(new_image[index, pixel_to_expand:pixel_to_expand + 2])
        # print('image shape: ', new_image[index, pixel_to_expand:].shape, 'big image shape: ',
        # big_image[index, pixel_to_expand + 1:].shape)
        big_image[index, pixel_to_expand + 1:] = new_image[index, pixel_to_expand:]
        big_image[index, pixel_to_expand + 1] = value_to_add
        if mask is not None:
            big_mask[index, pixel_to_expand + 1:] = mask[index, pixel_to_expand:]
            big_mask[index, pixel_to_expand] = 1
    return big_image, big_mask


def seam_carve(image, mode, mask=None):
    # if mask is not None:
    #    print(mode, 'with mask')
    # else:
    #    print(mode)

    new_image = image.copy()
    intensity = get_intensity(new_image)
    grad_intensity = get_grad_intensity(intensity)  # get_grad_intensity(intensity)

    if mask is not None:
        grad_intensity = get_intensity_with_mask(grad_intensity, mask)

    if 'vertical' in mode:
        grad_intensity = grad_intensity.T
        if mask is not None:
            mask = mask.T
        new_image = new_image.transpose((1, 0, 2))

    intensity_sum = get_intensities_sum(grad_intensity)
    min_seam = get_min_seam(intensity_sum)
    # seam_mask = get_seam_mask(min_seam, new_image.shape[:2])
    seam = get_min_seam(intensity_sum)

    # print(len(min_seam))
    # print(new_image.shape)

    # if 'shrink' in mode:
    #    new_image, mask = shrink_seam(new_image, mask, min_seam)
    # if 'expand' in mode:
    #    new_image, mask = expand_seam(new_image, mask, min_seam)

    if 'vertical' in mode:
       if mask is not None:
           mask = mask.T
       seam = seam.T
    #    # new_image = new_image.transpose((1, 0, 2))

    return image, mask, seam
