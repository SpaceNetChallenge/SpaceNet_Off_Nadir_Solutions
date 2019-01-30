import os
import numpy as np
from keras.preprocessing.image import Iterator
import skimage.io


class BaseMaskDatasetIterator(Iterator):
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
                 add_contours=True
                ):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_ids = image_ids
        self.image_name_template = image_name_template
        self.mask_name_template = mask_name_template
        self.random_transformer = random_transformer
        self.crop_shape = crop_shape
        self.preprocessing_function = preprocessing_function
        self.add_contours = add_contours
        super(BaseMaskDatasetIterator, self).__init__(len(self.image_ids), batch_size, shuffle, None)
    
    def pad_mask_image(self, mask, image, img_id, crop_shape):
        # _mask = mask[:, :, 0]
        return NotImplementedError

    def transform_batch_y(self, batch_y):
        return batch_y

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = []
        batch_y = []

        for batch_index, image_index in enumerate(index_array):
            _id = self.image_ids[image_index]
            # print(_id)
            img_name = self.image_name_template.format(id=_id)
            mask_name = self.mask_name_template.format(id=_id)

            image = skimage.io.imread(os.path.join(self.images_dir, img_name), plugin='tifffile')
            mask = skimage.io.imread(os.path.join(self.masks_dir, mask_name), plugin='tifffile')
            if self.add_contours:
                tmp = np.zeros(mask.shape[:2], dtype=mask.dtype)
                mask = np.dstack([mask, tmp])
            else:
                mask = np.dstack([mask]*3)

            crop_mask, crop_image = self.pad_mask_image(mask, image, _id, self.crop_shape)
            if self.random_transformer is not None:
                data = self.random_transformer(image=crop_image, mask=crop_mask)
                crop_image, crop_mask = data['image'], data['mask']

            if self.add_contours:
                crop_mask = crop_mask / 255.
                crop_mask = crop_mask[:, :, :2]
            else:
                crop_mask = crop_mask / 255.
                crop_mask = crop_mask[:, :, 0]
                crop_mask = np.expand_dims(crop_mask, -1)

            batch_x.append(crop_image)
            batch_y.append(crop_mask)
            
        batch_x = np.array(batch_x, dtype="float32")
        batch_y = np.array(batch_y, dtype="float32")
        if self.preprocessing_function:
            batch_x = self.preprocess_input(batch_x)

        return self.transform_batch_x(batch_x), self.transform_batch_y(batch_y)

    @staticmethod
    def preprocess_input(batch_x):
        means = [963.0, 805.0, 666.0]
        stds = [473.0, 403.0, 395.0]
        for i in range(batch_x.shape[3]):
            batch_x[:, :, :, i] = (batch_x[:, :, :, i] - means[i]) / stds[i]
        return batch_x

    def transform_batch_x(self, batch_x):
        return batch_x

    def next(self):

        with self.lock:
            index_array = next(self.index_generator)
        return self._get_batches_of_transformed_samples(index_array)


