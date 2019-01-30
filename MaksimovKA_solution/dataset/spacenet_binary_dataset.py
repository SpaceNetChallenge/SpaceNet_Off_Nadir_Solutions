import numpy as np
import pandas as pd
from .base_dataset import BaseMaskDatasetIterator
from albumentations import RandomCrop, Compose, PadIfNeeded

class SpacenetBinaryDataset:
    def __init__(self,
                 images_dir,
                 masks_dir,
                 folds_file,
                 fold=0,
                 fold_num=5,
                 add_contours=False
                 ):
        super().__init__()
        self.fold = fold
        self.folds_file = folds_file
        self.fold_num = fold_num
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.add_contours = add_contours
        self.train_ids, self.val_ids = self.generate_ids()
        print("Found {} train images".format(len(self.train_ids)))
        print("Found {} val images".format(len(self.val_ids)))

    def get_generator(self, image_ids, crop_shape, preprocessing_function=1,
                      random_transformer=None, batch_size=16, shuffle=True):
        return SpacenetDatasetIterator(
            self.images_dir,
            self.masks_dir,
            image_ids,
            crop_shape,
            preprocessing_function,
            random_transformer,
            batch_size,
            shuffle=shuffle,
            image_name_template="{id}.tif",
            mask_name_template="{id}.tif",
            add_contours=self.add_contours
        )

    def train_generator(self, crop_shape, preprocessing_function=1, random_transformer=None, batch_size=16):
        return self.get_generator(self.train_ids, crop_shape, preprocessing_function,
                                  random_transformer, batch_size, True)

    def val_generator(self, preprocessing_function=1, batch_size=1):
        return self.get_generator(self.val_ids, (928, 928), preprocessing_function, None, batch_size, False)

    def generate_ids(self):
        
        df = pd.read_csv(self.folds_file)
        
        df['angle'] = df['img_id'].apply(lambda x: int(x.split('_')[1][5:]))
        # df = df[(df['angle'] > 25) & (df['angle'] <= 40)]
        # df = df[df['angle'] > 40]
        val_ids = df[(df['fold_on_train'] == self.fold)]['img_id'].values
        train_ids = np.sort(df[(df['fold_on_predict'] != self.fold)]['img_id'].values)
        return train_ids, val_ids


class SpacenetDatasetIterator(BaseMaskDatasetIterator):

    def __init__(self,
                 images_dir,
                 masks_dir,
                 image_ids,
                 crop_shape,
                 preprocessing_function,
                 random_transformer=None,
                 batch_size=8,
                 shuffle=True,
                 image_name_template=None,
                 mask_name_template=None,
                 add_contours=False):
        super().__init__(images_dir,
                         masks_dir,
                         image_ids,
                         crop_shape,
                         preprocessing_function,
                         random_transformer,
                         batch_size,
                         shuffle,
                         image_name_template,
                         mask_name_template,
                         add_contours)

    def pad_mask_image(self, mask, image, img_id, crop_shape):
        composed = Compose([PadIfNeeded(crop_shape[0], crop_shape[1], p=1),
                            RandomCrop(crop_shape[0], crop_shape[1], p=1)], p=1)

        if np.sum(mask) != 0:

            s = 0
            tries = 0
            while s == 0:
                # crop = composed(crop_shape[0], crop_shape[1])
                croped = composed(image=image, mask=mask)

                image_padded = croped['image']
                mask_padded = croped['mask']
                # print(mask_padded.shape)
                s = np.sum(mask_padded)
                tries += 1
                if tries > 5:
                    break
        else:

            croped = composed(image=image, mask=mask)
            image_padded = croped['image']
            mask_padded = croped['mask']
            
        return mask_padded, image_padded
