import os

import cv2
import numpy as np
import pandas as pd
from skimage.external import tifffile

cv2.setNumThreads(0)
from torch.utils.data import Dataset

nadirs = {
    "7-8": ["Atlanta_nadir7_catid_1030010003D22F00", "Atlanta_nadir8_catid_10300100023BC100"],
    "10": ["Atlanta_nadir10_catid_1030010003993E00", "Atlanta_nadir10_catid_1030010003CAF100"],

    "13-16": ["Atlanta_nadir13_catid_1030010002B7D800", "Atlanta_nadir14_catid_10300100039AB000",
              "Atlanta_nadir16_catid_1030010002649200"],

    "19-23": ["Atlanta_nadir19_catid_1030010003C92000", "Atlanta_nadir21_catid_1030010003127500",
              "Atlanta_nadir23_catid_103001000352C200"],

    "25-29": ["Atlanta_nadir25_catid_103001000307D800", "Atlanta_nadir27_catid_1030010003472200",
              "Atlanta_nadir29_catid_1030010003315300"],
    "30-34": ["Atlanta_nadir30_catid_10300100036D5200", "Atlanta_nadir32_catid_103001000392F600",
              "Atlanta_nadir34_catid_1030010003697400"],
    "36-39": ["Atlanta_nadir36_catid_1030010003895500", "Atlanta_nadir39_catid_1030010003832800"],
    "42-44": ["Atlanta_nadir42_catid_10300100035D1B00", "Atlanta_nadir44_catid_1030010003CCD700"],
    "46-47": ["Atlanta_nadir46_catid_1030010003713C00", "Atlanta_nadir47_catid_10300100033C5200"],
    "49-50": ["Atlanta_nadir49_catid_1030010003492700", "Atlanta_nadir50_catid_10300100039E6200"],
    "52-53": ["Atlanta_nadir52_catid_1030010003BDDC00", "Atlanta_nadir53_catid_1030010003193D00",
              "Atlanta_nadir53_catid_1030010003CD4300"],

    "0-25": ["Atlanta_nadir7_catid_1030010003D22F00", "Atlanta_nadir8_catid_10300100023BC100",
             "Atlanta_nadir10_catid_1030010003993E00", "Atlanta_nadir10_catid_1030010003CAF100",
             "Atlanta_nadir13_catid_1030010002B7D800", "Atlanta_nadir14_catid_10300100039AB000",
             "Atlanta_nadir16_catid_1030010002649200",
             "Atlanta_nadir19_catid_1030010003C92000", "Atlanta_nadir21_catid_1030010003127500",
             "Atlanta_nadir23_catid_103001000352C200", "Atlanta_nadir25_catid_103001000307D800"],

    "26-40": ["Atlanta_nadir27_catid_1030010003472200", "Atlanta_nadir29_catid_1030010003315300",
              "Atlanta_nadir30_catid_10300100036D5200", "Atlanta_nadir32_catid_103001000392F600",
              "Atlanta_nadir34_catid_1030010003697400", "Atlanta_nadir36_catid_1030010003895500",
              "Atlanta_nadir39_catid_1030010003832800"],
    "41-55": ["Atlanta_nadir42_catid_10300100035D1B00", "Atlanta_nadir44_catid_1030010003CCD700",
              "Atlanta_nadir46_catid_1030010003713C00", "Atlanta_nadir47_catid_10300100033C5200",
              "Atlanta_nadir49_catid_1030010003492700", "Atlanta_nadir50_catid_10300100039E6200",
              "Atlanta_nadir52_catid_1030010003BDDC00", "Atlanta_nadir53_catid_1030010003193D00",
              "Atlanta_nadir53_catid_1030010003CD4300"],

    "all": ["Atlanta_nadir7_catid_1030010003D22F00", "Atlanta_nadir8_catid_10300100023BC100",
            "Atlanta_nadir10_catid_1030010003993E00",
            "Atlanta_nadir10_catid_1030010003CAF100",
            "Atlanta_nadir13_catid_1030010002B7D800", "Atlanta_nadir14_catid_10300100039AB000",
            "Atlanta_nadir16_catid_1030010002649200",
            "Atlanta_nadir19_catid_1030010003C92000",
            "Atlanta_nadir21_catid_1030010003127500", "Atlanta_nadir23_catid_103001000352C200",
            "Atlanta_nadir25_catid_103001000307D800",
            "Atlanta_nadir27_catid_1030010003472200",
            "Atlanta_nadir29_catid_1030010003315300", "Atlanta_nadir30_catid_10300100036D5200",
            "Atlanta_nadir32_catid_103001000392F600",
            "Atlanta_nadir34_catid_1030010003697400",
            "Atlanta_nadir36_catid_1030010003895500", "Atlanta_nadir39_catid_1030010003832800",
            "Atlanta_nadir42_catid_10300100035D1B00",
            "Atlanta_nadir44_catid_1030010003CCD700",
            "Atlanta_nadir46_catid_1030010003713C00", "Atlanta_nadir47_catid_10300100033C5200",
            "Atlanta_nadir49_catid_1030010003492700",
            "Atlanta_nadir50_catid_10300100039E6200",
            "Atlanta_nadir52_catid_1030010003BDDC00", "Atlanta_nadir53_catid_1030010003193D00",
            "Atlanta_nadir53_catid_1030010003CD4300"]

}


def stretch_8bit(bands, lower_percent=0, higher_percent=100):
    out = np.zeros_like(bands).astype(np.float32)
    for i in range(bands.shape[-1]):
        a = 0
        b = 1
        band = bands[:, :, i].flatten()
        filtered = band[band > 0]
        if len(filtered) == 0:
            continue
        c = np.percentile(filtered, lower_percent)
        d = np.percentile(filtered, higher_percent)
        t = a + (bands[:, :, i] - c) * (b - a) / (d - c)
        t[t < a] = a
        t[t > b] = b
        out[:, :, i] = t
    return out.astype(np.float32)


class DenseData(Dataset):
    def __init__(self, data_path, nadir, mode="train", csv_path="folds.csv", fold=0, transform=None):
        super().__init__()
        self.nadir = nadir
        self.data_path = data_path
        names = []
        df = pd.read_csv(csv_path)
        if mode == "train":
            ids = set(df[(df['fold'] != fold)]['id'].tolist())
        else:
            ids = set(df[(df['fold'] == fold)]['id'].tolist())

        for cat in nadirs[nadir]:
            if os.path.exists(os.path.join(self.data_path, cat, "Pan-Sharpen")):
                names.extend(os.path.join(cat, "Pan-Sharpen", f) for f in
                             os.listdir(os.path.join(self.data_path, cat, "Pan-Sharpen")) if f in ids)

        self.names = names
        self.transform = transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        try:
            img = tifffile.imread(os.path.join(self.data_path, self.names[idx]))
        except Exception as e:
            print(os.path.join(self.data_path, self.names[idx]))
            raise e
        if np.shape(img)[0] == 4:
            img = np.moveaxis(img, 0, -1)
        img = stretch_8bit(img)

        mask = cv2.imread(os.path.join("train_labels", "masks_all",
                                       "mask_" + "_".join(self.names[idx][:-4].split("_")[-2:]) + ".png"),
                          cv2.IMREAD_COLOR)
        if mask is None:
            mask = []
        else:
            mask = mask / 255.
        nadir = nadirs['all'].index(self.names[idx].split("/")[0])
        angle = np.zeros((1, 1, 27))
        angle[0, 0, nadir] = 1
        sample = {"img": img, "mask": mask, 'img_name': self.names[idx], "angle": angle}
        if self.transform:
            sample = self.transform(sample)
        return sample


class TestDenseData(Dataset):
    def __init__(self, data_path, transform=None):
        super().__init__()

        self.data_path = data_path
        self.names = [os.path.join(data_path, "Pan-Sharpen", f) for f in
                      os.listdir(os.path.join(self.data_path, "Pan-Sharpen")) if f.endswith("tif")]
        self.transform = transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        try:
            img = tifffile.imread(os.path.join(self.data_path, self.names[idx]))
        except Exception as e:
            print(os.path.join(self.data_path, self.names[idx]))
            raise e
        if np.shape(img)[0] == 4:
            img = np.moveaxis(img, 0, -1)
        img = stretch_8bit(img)
        mask = []
        nadir = nadirs['all'].index(self.names[idx].split("/")[-3])
        angle = np.zeros((1, 1, 27))
        angle[0, 0, nadir] = 1
        sample = {"img": img, "mask": mask, 'img_name': self.names[idx], "angle": angle}
        if self.transform:
            sample = self.transform(sample)
        return sample
