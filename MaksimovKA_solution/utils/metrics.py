import keras.backend as K
import numpy as np


def hard_dice_coef_mask(y_true, y_pred, smooth=1e-3):
    y_true_f = K.flatten(K.round(y_true[..., 0]))
    y_pred_f = K.flatten(K.round(y_pred[..., 0]))
    intersection = K.sum(y_true_f * y_pred_f)
    return 100. * (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def hard_jacard_coef_mask(y_true, y_pred, smooth=1e-3):
    # K.flatten(K.round(y_true[..., 0]))
    y_true_f = K.flatten(K.round(y_true[..., 0]))
    y_pred_f =K.flatten(K.round(y_pred[..., 0]))
    intersection = K.sum(y_true_f * y_pred_f)
    return 100.0 * (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)


def hard_dice_coef_border(y_true, y_pred, smooth=1e-3):
    y_true_f = K.flatten(K.round(y_true[..., 1]))
    y_pred_f = K.flatten(K.round(y_pred[..., 1]))
    intersection = K.sum(y_true_f * y_pred_f)
    return 100. * (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def hard_jacard_coef_border(y_true, y_pred, smooth=1e-3):
    # K.flatten(K.round(y_true[..., 0]))
    y_true_f = K.flatten(K.round(y_true[..., 1]))
    y_pred_f =K.flatten(K.round(y_pred[..., 1]))
    intersection = K.sum(y_true_f * y_pred_f)
    return 100.0 * (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)


def calc_iou(gt_masks, predicted_masks, height=768, width=768):
    true_objects = gt_masks.shape[2]
    pred_objects = predicted_masks.shape[2]
    labels = np.zeros((height, width), np.uint16)
    
    for index in range(0, true_objects):
        labels[gt_masks[:, :, index] > 0] = index + 1
    y_true = labels.flatten()
    labels_pred = np.zeros((height, width), np.uint16)
    
    for index in range(0, pred_objects):
        if sum(predicted_masks[:, :, index].shape) == height + width:
            labels_pred[predicted_masks[:, :, index] > 0] = index + 1
    y_pred = labels_pred.flatten()
    
    intersection = np.histogram2d(y_true, y_pred, bins=(true_objects + 1, pred_objects + 1))[0]
    
    area_true = np.histogram(labels, bins=true_objects + 1)[0]
    area_pred = np.histogram(y_pred, bins=pred_objects + 1)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)
    
    # Compute union
    union = area_true + area_pred - intersection
    
    # Exclude background from the analysis
    intersection = intersection[1:, 1:]
    union = union[1:, 1:]
    union[union == 0] = 1e-9
    # print(union)
    # print(intersection)
    iou = intersection / union
    return iou


def precision_at(threshold, iou):
    matches = iou > threshold
    true_positives = np.sum(matches, axis=1) == 1  # Correct objects
    false_positives = np.sum(matches, axis=0) == 0  # Missed objects
    false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
    tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
    return tp, fp, fn

