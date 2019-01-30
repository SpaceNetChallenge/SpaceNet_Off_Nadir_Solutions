import numpy as np


def multi_rle_encode(labels):
    return [rle_encode(labels == k) for k in np.unique(labels[labels > 0])]


def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T


def masks_as_image(in_mask_list, all_masks=None, shape=(768, 768)):
    if all_masks is None:
        all_masks = np.zeros(shape, dtype=np.int16)
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks += rle_decode(mask)
    return np.expand_dims(all_masks, -1)


def masks_as_label(in_mask_list, all_masks=None, shape=(768, 768)):
    if all_masks is None:
        all_masks = np.zeros(shape, dtype=np.int16)
    ship_label = 0
    for mask in in_mask_list:
        if isinstance(mask, str):
            ship_label += 1
            all_masks += ship_label * rle_decode(mask)
    return np.expand_dims(all_masks, -1)
