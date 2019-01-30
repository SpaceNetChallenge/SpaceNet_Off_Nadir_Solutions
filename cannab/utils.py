import numpy as np

cat_ids = ['10300100023BC100', '1030010002649200', '1030010002B7D800', '103001000307D800', '1030010003127500', '1030010003193D00', '1030010003315300', '10300100033C5200', '1030010003472200', '1030010003492700',
 '103001000352C200', '10300100035D1B00', '1030010003697400', '10300100036D5200', '1030010003713C00', '1030010003832800', '1030010003895500', '103001000392F600', '1030010003993E00', '10300100039AB000', '10300100039E6200',
 '1030010003BDDC00', '1030010003C92000', '1030010003CAF100', '1030010003CCD700', '1030010003CD4300', '1030010003D22F00']

cat_nadirs = [8, 16, 13, 25, 21, 53, 29, 47, 27, 49, 23, 42, 34, 30, 46, 39, 36, 32, 10, 14, 50, 52, 19, 10, 44, 53, 7]

def parse_img_id(img_id):
    tmp = img_id.split('_')
    nadir = int(tmp[1].split('nadir')[1])

    cid = 0 
    try:
        cid = cat_ids.index(tmp[3])
    except:
        bst_dif = 10000
        for i in range(len(cat_nadirs)):
            if abs(cat_nadirs[i] - nadir) < bst_dif:
                bst_dif = abs(cat_nadirs[i] - nadir) 
                cid = i

    cat_inp = np.zeros((27,), dtype='float')
    cat_inp[cid] = 1
    
    x_coord = 740801
    try:
        x_coord = int(tmp[4])
    except:
        x_coord = 740801
    if x_coord < 732701:
        x_coord = 732701
    if x_coord > 748451:
        x_coord = 748451

    y_coord = 3731889
    try:
        y_coord = int(tmp[5].split('.png')[0])
    except:
        y_coord = 3731889
    if y_coord < 3720189:
        y_coord = 3720189
    if y_coord > 3743589:
        y_coord = 3743589

    y_coord -= 3710000
    y_coord /= 40000
    x_coord -= 720000
    x_coord /= 40000

    return nadir, cat_inp, np.asarray([x_coord, y_coord])


def preprocess_inputs(x):
    x = np.asarray(x, dtype='float32')
    x /= 127
    x -= 1
    return x

def dice(im1, im2, empty_score=1.0):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        Both are empty (sum eq to zero) = empty_score

    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum