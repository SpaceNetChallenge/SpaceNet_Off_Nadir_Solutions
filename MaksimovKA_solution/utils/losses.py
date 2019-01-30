import sys; sys.path.append('..')
import keras.backend as K

def binary_crossentropy(y, p):
    return K.mean(K.binary_crossentropy(y, p))


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return K.mean((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def dice_coef_loss_bce(y_true, y_pred, dice=0.5, bce=0.5):
    return binary_crossentropy(y_true, y_pred) * bce + dice_coef_loss(y_true, y_pred) * dice
    

def jacard_coef(y_true, y_pred, smooth=1e-3):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)


def jacard_coef_loss(y_true, y_pred):
    return 1 - jacard_coef(y_true, y_pred)


def double_head(y_true, y_pred, instance=1.0, border=1.0):
    mask_loss = dice_coef_loss_bce(y_true[..., 0], y_pred[..., 0]) * instance
    contour_loss = dice_coef_loss_bce(y_true[..., 1], y_pred[..., 1]) * border
    return mask_loss + contour_loss

def double_head_changed(y_true, y_pred, instance=1.0, border=1.0):
    mask_loss = dice_coef_loss_bce(y_true[..., 0], y_pred[..., 0], dice=0.3, bce=0.7) * instance
    contour_loss = dice_coef_loss_bce(y_true[..., 1], y_pred[..., 1], dice=0.3, bce=0.7) * border
    return mask_loss + contour_loss


def make_loss(loss_name):
    if loss_name == 'bce':
        def loss(y, p):
            return binary_crossentropy(y, p)
        return loss
    elif loss_name == 'bce_dice':
        def loss(y, p):
            return dice_coef_loss_bce(y, p, dice=0.5, bce=0.5)
    
        return loss
    elif loss_name == 'bce_jaccard':
        def loss(y, p):
            return 0.5*binary_crossentropy(y, p) + 0.5*jacard_coef_loss(y, p)
        return loss


    elif loss_name == 'double_head':
        def loss(y, p):
            return double_head(y, p, instance=1.0, border=1.0)
    
        return loss
    elif loss_name == 'double_head_changed':
        def loss(y, p):
            return double_head_changed(y, p, instance=1.0, border=1.0)

        return loss
    else:
        ValueError("Unknown loss.")