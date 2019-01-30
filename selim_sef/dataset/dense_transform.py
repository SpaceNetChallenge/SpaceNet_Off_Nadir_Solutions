import math
import random

import cv2

cv2.setNumThreads(0)

import numpy as np
import torch
from numpy.core.multiarray import ndarray

_DEFAULT_ALPHASTD = 0.1
_DEFAULT_EIGVAL = torch.Tensor([0.2175, 0.0188, 0.0045])
_DEFAULT_EIGVEC = torch.Tensor([[-0.5675, 0.7192, 0.4009], [-0.5808, -0.0045, -0.8140], [-0.5836, -0.6948, 0.4203]])
_DEFAULT_BCS = [0.2, 0.2, 0.2]


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        sample["img"] = self.normalize(sample["img"], self.mean, self.std)
        if "imgs" in sample:
            sample["imgs"] = [self.normalize(img, self.mean, self.std) for img in sample["imgs"]]
        return sample

    def normalize(self, tensor, mean, std):
        if not (torch.is_tensor(tensor) and tensor.ndimension() == 3):
            raise TypeError('tensor is not a torch image.')
        for t, m, s in zip(tensor, mean, std):
            t.sub_(m).div_(s)
        return tensor


class HFlip(object):
    def __call__(self, sample):
        if random.random() < 0.5:
            try:
                sample["img"] = cv2.flip(sample["img"], 1)
            except Exception as e:
                print(sample["img_name"])
                raise e

            if sample["mask"] is not None:
                sample["mask"] = cv2.flip(sample["mask"], 1)
        return sample


class VFlip(object):
    def __call__(self, sample):
        if random.random() < 0.5:
            sample["img"] = cv2.flip(sample["img"], 0)
            if sample["mask"] is not None:
                sample["mask"] = cv2.flip(sample["mask"], 0)
        return sample


def rot90(img, factor):
    img = np.rot90(img, factor)
    return np.ascontiguousarray(img)


class Rotate90(object):
    def __call__(self, sample):
        if random.random() < 0.5:
            factor = random.randint(0, 4)
            sample["img"] = rot90(sample["img"], factor)
            if sample["mask"] is not None:
                sample["mask"] = rot90(sample["mask"], factor)
        return sample

class Pad(object):
    def __init__(self, block=32, mode='reflect'):
        super().__init__()
        self.block = block
        self.mode = mode

    def __call__(self, sample):
        sample["img"] = pad(sample["img"], self.block, type='reflect')
        if sample["mask"] is not None and sample["mask"] != []:
            sample["mask"] = pad(sample["mask"], self.block, type='reflect')
        return sample


def pad(image, block, type='reflect', **kwargs):
    params = {}
    if type == 'zero':
        params = {'constant_values': 0}
        type = 'constant'
    x0, x1, y0, y1 = 0, 0, 0, 0
    if (image.shape[1] % block) != 0:
        x0 = int((block - image.shape[1] % block) / 2)
        x1 = (block - image.shape[1] % block) - x0
    if (image.shape[0] % block) != 0:
        y0 = int((block - image.shape[0] % block) / 2)
        y1 = (block - image.shape[0] % block) - y0
    if len(image.shape) > 1:
        return np.pad(image, ((y0, y1), (x0, x1), (0, 0)), type, **params, **kwargs)
    else:
        return np.pad(image, ((y0, y1), (x0, x1)), type, **params, **kwargs)

class ToTensor(object):
    def __call__(self, sample):
        sample["img"] = torch.from_numpy(sample["img"].transpose((2, 0, 1))).float()
        sample["angle"] = torch.from_numpy(sample["angle"].transpose((2, 0, 1))).float()
        if isinstance(sample["mask"], ndarray):
            sample["mask"] = torch.from_numpy(sample["mask"].transpose((2, 0, 1))).float()
        return sample


class ColorJitterImage(object):
    def __init__(self):
        self.transform = ColorJitter()

    def __call__(self, sample):
        if random.random() < 0.5:
            sample["img"] = self.transform(sample['img'])
        return sample


class LightingImage(object):
    def __init__(self):
        self.transform = Lighting()

    def __call__(self, sample):
        if random.random() < 0.5:
            sample["img"] = self.transform(sample['img'])
        return sample

class RandomCropAndScale(object):
    def __init__(self, height, width, scale_range=(0.5, 2.0), rescale_prob=0.5, prob=1.):
        self.prob = prob
        self.height = height
        self.width = width
        self.scale_range = scale_range
        self.rescale_prob = rescale_prob

    def __call__(self, sample):
        if random.random() > self.prob:
            return sample
        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        if random.random() > self.rescale_prob:
            scale = 1.
        random_state = np.random.randint(0, 10000)

        sample["img"] = random_crop(sample['img'], self.height, self.width, scale, np.random.RandomState(random_state))
        if sample["mask"] is not None and sample["mask"] != []:
            sample["mask"] = random_crop(sample['mask'], self.height, self.width, scale,
                                         np.random.RandomState(random_state), mode=cv2.INTER_NEAREST)
        return sample

def random_crop(img, height, width, scale, random_state, mode=None):
    if random_state is None:
        random_state = np.random.RandomState(1234)
    crop_height = height
    crop_width = width
    img_height, img_width = img.shape[:2]

    max_height = int(min(crop_height * scale, img_height))
    max_width = int(min(crop_width * scale, img_width))
    adjusted_scale = scale * min(max_width / (crop_width * scale), max_height / (crop_height * scale))
    crop_width = int(adjusted_scale * width)
    crop_height = int(adjusted_scale * height)

    start_y = random_state.randint(0, max(img_height - crop_height, 1))
    start_x = random_state.randint(0, max(img_width - crop_width, 1))
    crop = img[start_y:start_y + crop_height, start_x:start_x + crop_width]
    if mode is None:
        if 1 / adjusted_scale < 1.:
            mode = cv2.INTER_AREA
        else:
            mode = cv2.INTER_CUBIC
    if scale != 1.:
        img = cv2.resize(crop, (width, height), interpolation=mode)
    else:
        img = crop
    return img



def shift_scale_rotate(img, angle, scale, dx, dy, borderMode=cv2.BORDER_CONSTANT):
    height, width = img.shape[:2]

    cc = math.cos(angle / 180 * math.pi) * scale
    ss = math.sin(angle / 180 * math.pi) * scale
    rotate_matrix = np.array([[cc, -ss], [ss, cc]])

    box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
    box1 = box0 - np.array([width / 2, height / 2])
    box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx * width, height / 2 + dy * height])

    box0 = box0.astype(np.float32)
    box1 = box1.astype(np.float32)
    mat = cv2.getPerspectiveTransform(box0, box1)
    img = cv2.warpPerspective(img, mat, (width, height),
                              flags=cv2.INTER_NEAREST,
                              borderMode=borderMode)

    return img


class RandomRotate(object):
    def __init__(self, angle=15, prob=0.3):
        self.prob = prob
        self.angle = angle


    def __call__(self, sample):
        if random.random() > self.prob:
            return sample
        angle = random.uniform(-self.angle, self.angle)
        if angle == 0:
            return sample

        sample["img"] = shift_scale_rotate(sample['img'], angle=angle, scale=1, dx=0, dy=0)
        if sample["mask"] is not None and sample["mask"] != []:
            sample["mask"] = shift_scale_rotate(sample['mask'], angle=angle, scale=1, dx=0, dy=0)
        return sample

def _grayscale(img):
    alpha = torch.Tensor([0.299, 0.587, 0.114])
    return (alpha.view(3, 1, 1) * img).sum(0, keepdim=True)


def _blend(img1, img2, alpha):
    return img1 * alpha + (1 - alpha) * img2


class Lighting(object):
    def __init__(self, alphastd=_DEFAULT_ALPHASTD, eigval=_DEFAULT_EIGVAL, eigvec=_DEFAULT_EIGVEC):
        self._alphastd = alphastd
        self._eigval = eigval
        self._eigvec = eigvec

    def __call__(self, img):
        if self._alphastd == 0.:
            return img

        alpha = torch.normal(torch.zeros(3), self._alphastd)
        rgb = (self._eigvec * alpha * self._eigval).sum(dim=1)
        return img + rgb.view(3, 1, 1)


class Saturation(object):
    def __init__(self, var):
        self._var = var

    def __call__(self, img):
        gs = _grayscale(img)
        alpha = torch.FloatTensor(1).uniform_(-self._var, self._var) + 1.0
        return _blend(img, gs, alpha)


class Brightness(object):
    def __init__(self, var):
        self._var = var

    def __call__(self, img):
        gs = torch.zeros(img.size())
        alpha = torch.FloatTensor(1).uniform_(-self._var, self._var) + 1.0
        return _blend(img, gs, alpha)


class Contrast(object):
    def __init__(self, var):
        self._var = var

    def __call__(self, img):
        gs = _grayscale(img)
        gs = torch.FloatTensor(1, 1, 1).fill_(gs.mean())
        alpha = torch.FloatTensor(1).uniform_(-self._var, self._var) + 1.0
        return _blend(img, gs, alpha)


class ColorJitter(object):
    def __init__(self, saturation=_DEFAULT_BCS[0], brightness=_DEFAULT_BCS[1], contrast=_DEFAULT_BCS[2]):
        self._transforms = []
        if saturation is not None:
            self._transforms.append(Saturation(saturation))
        if brightness is not None:
            self._transforms.append(Brightness(brightness))
        if contrast is not None:
            self._transforms.append(Contrast(contrast))

    def __call__(self, img):
        if len(self._transforms) == 0:
            return img

        for t in random.sample(self._transforms, len(self._transforms)):
            img[:3, ...] = t(img[:3,...])

        return img
