import cv2

import numpy as np
from torch.nn import functional as F


class TTAOp:
    def __init__(self, sigmoid=True):
        self.sigmoid = sigmoid

    def __call__(self, model, batch):
        forwarded = self.forward(batch[0])

        predicted = model((torch.from_numpy(forwarded).cuda().float(), batch[1]))
        return self.backward(self.to_numpy(predicted))

    def forward(self, img):
        raise NotImplementedError

    def backward(self, img):
        raise NotImplementedError

    def to_numpy(self, batch):
        if self.sigmoid:
            batch = F.sigmoid(batch)
        else:
            batch = F.softmax(batch, dim=1)
        data = batch.data.cpu().numpy()
        return data


class BasicTTAOp(TTAOp):
    @staticmethod
    def op(img):
        raise NotImplementedError

    def forward(self, img):
        return self.op(img)

    def backward(self, img):
        if len(img.shape) == 4:
            # segmentation
            return self.forward(img)
        else:
            # classification
            return img


class Nothing(BasicTTAOp):
    @staticmethod
    def op(img):
        return img


class ScaleUp2(BasicTTAOp):
    @staticmethod
    def op(img):
        batch_size = img.shape[0]
        channels = img.shape[1]
        out = np.zeros((batch_size, channels, 1312, 1312))
        for i in range(batch_size):
            in_img = np.moveaxis(img[i], 0, -1)
            out_img = cv2.resize(in_img, (1312, 1312), interpolation=cv2.INTER_CUBIC)
            out[i] = np.moveaxis(out_img, -1, 0)
        return out

    def backward(self, img):
        batch_size = img.shape[0]
        channels = img.shape[1]
        out = np.zeros((batch_size, channels, 928, 928))
        for i in range(batch_size):
            in_img = np.moveaxis(img[i], 0, -1)
            out_img = cv2.resize(in_img, (928, 928), interpolation=cv2.INTER_AREA)
            out[i] = np.moveaxis(out_img, -1, 0)
        return out

class ScaleUp1(BasicTTAOp):
    @staticmethod
    def op(img):
        batch_size = img.shape[0]
        channels = img.shape[1]
        out = np.zeros((batch_size, channels, 1120, 1120))
        for i in range(batch_size):
            in_img = np.moveaxis(img[i], 0, -1)
            out_img = cv2.resize(in_img, (1120, 1120), interpolation=cv2.INTER_CUBIC)
            out[i] = np.moveaxis(out_img, -1, 0)
        return out

    def backward(self, img):
        batch_size = img.shape[0]
        channels = img.shape[1]
        out = np.zeros((batch_size, channels, 928, 928))
        for i in range(batch_size):
            in_img = np.moveaxis(img[i], 0, -1)
            out_img = cv2.resize(in_img, (928, 928), interpolation=cv2.INTER_AREA)
            out[i] = np.moveaxis(out_img, -1, 0)
        return out

class ScaleDown(BasicTTAOp):
    @staticmethod
    def op(img):
        batch_size = img.shape[0]
        channels = img.shape[1]
        out = np.zeros((batch_size, channels, 768, 768))
        for i in range(batch_size):
            in_img = np.moveaxis(img[i], 0, -1)
            out_img = cv2.resize(in_img, (768, 768), interpolation=cv2.INTER_AREA)
            out[i] = np.moveaxis(out_img, -1, 0)
        return out

    def backward(self, img):
        batch_size = img.shape[0]
        channels = img.shape[1]
        out = np.zeros((batch_size, channels, 928, 928))
        for i in range(batch_size):
            in_img = np.moveaxis(img[i], 0, -1)
            out_img = cv2.resize(in_img, (928, 928), interpolation=cv2.INTER_CUBIC)
            out[i] = np.moveaxis(out_img, -1, 0)
        return out


class HFlip(BasicTTAOp):
    @staticmethod
    def op(img):
        return np.ascontiguousarray(np.flip(img, axis=-1))


class VFlip(BasicTTAOp):
    @staticmethod
    def op(img):
        return np.ascontiguousarray(np.flip(img, axis=-2))


class Transpose(BasicTTAOp):
    @staticmethod
    def op(img):
        return np.ascontiguousarray(img.transpose(0, 1, 3, 2))


def chain_op(data, operations):
    for op in operations:
        data = op.op(data)
    return data


class ChainedTTA(TTAOp):
    @property
    def operations(self):
        raise NotImplementedError

    def forward(self, img):
        return chain_op(img, self.operations)

    def backward(self, img):
        if len(img.shape) == 4:
            # segmentation
            return chain_op(img, reversed(self.operations))
        else:
            # classification
            return img


class HVFlip(ChainedTTA):
    @property
    def operations(self):
        return [HFlip, VFlip]


class TransposeHFlip(ChainedTTA):
    @property
    def operations(self):
        return [Transpose, HFlip]


class TransposeVFlip(ChainedTTA):
    @property
    def operations(self):
        return [Transpose, VFlip]


class TransposeHVFlip(ChainedTTA):
    @property
    def operations(self):
        return [Transpose, HFlip, VFlip]


transforms2 = [Nothing, HFlip]
transforms8 = [Nothing, HFlip, VFlip, HVFlip, Transpose, TransposeHFlip, TransposeVFlip, TransposeHVFlip]
transforms4 = [Nothing, HFlip, VFlip, HVFlip]
transforms1 = [Nothing]
transforms_ms = [Nothing, ScaleUp1, ScaleUp2, ScaleDown]
transforms_ms_oof = [Nothing, ScaleUp1, ScaleDown]

import torch

print(torch.__version__)
