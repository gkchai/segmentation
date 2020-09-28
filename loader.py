# author: kcgarikipati@gmail.com

import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
import os
import utils.general

utils.general.silence_tf_logger()
logger = utils.general.get_logger("loader")


class Loader:
    """data loader class"""
    def __init__(self, cfg):
        self.cfg = cfg

    def _augmentation(self, image, weight, mask):
        """Returns (maybe) augmented images"""
        # how to flip two images together? use mask as additional channel
        # works since flip operations work on first two dimensions only
        aug_data = tf.concat([image, mask], axis=-1)
        if 'flip_h' in self.cfg.augment:
            aug_data = tf.image.random_flip_left_right(aug_data, self.cfg.seed)
        if 'flip_v' in self.cfg.augment:
            aug_data = tf.image.random_flip_up_down(aug_data, self.cfg.seed)
        # separate back image and mask
        image = aug_data[:, :, :-1]
        mask = aug_data[:, :, -1:]
        # other augmentations apply to image only
        if 'brightness' in self.cfg.augment:
            image = tf.image.random_brightness(image, max_delta=32.0 / 255.0, seed=self.cfg.seed)
        if 'hue' in self.cfg.augment:
            image = tf.image.random_hue(image, 0.3, seed=self.cfg.seed)
        if 'saturation' in self.cfg.augment:
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5, seed=self.cfg.seed)
        return image, weight, mask

    def preprocess(self, image_bytes, image_format='png', num_channels=3, crop=False):
        # decode bytes to RGB
        if image_format == 'png':
            image = tf.image.decode_png(image_bytes, channels=num_channels)
        elif image_format == 'jpeg':
            image = tf.image.decode_jpeg(image_bytes, channels=num_channels)
        else:
            raise ValueError("Unsupported format")

        # this applies pixel scaling to [0, 1]
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

        # Crop the central region of the image with an area containing 87.5% of
        # the original image.
        if crop:
            image = tf.image.central_crop(image, central_fraction=0.875)

        # resize_bilinear takes 4D-tensor so use w/ batch size 1
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_bilinear(
            image, [self.cfg.height, self.cfg.width], align_corners=False)
        image = tf.squeeze(image, [0])
        # (height, width, channels)
        return image

    def postprocess(self, image, dst_height, dst_width):
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_bilinear(
            image, [dst_height, dst_width], align_corners=False)
        image = tf.squeeze(image, [0])
        return image

    def _map_fn(self, image_path, weight, mask_path):

        image_bytes = tf.read_file(image_path)
        mask_bytes = tf.read_file(mask_path)

        image = self.preprocess(image_bytes, 'png', 3)
        mask = self.preprocess(mask_bytes, 'png', 1)

        return image, weight, mask

    def _create_data_ops(self, files_csv, is_train=True):

        logger.info("Loading files {}".format(files_csv))
        # can use tf.contrib.data.make_csv_dataset() but
        # issue is it already applies dataset.batch(1) internally
        # so per example map function can't be applied after batching

        # get the paths of images and masks
        csv_data = pd.DataFrame()
        for file_csv in files_csv:
            csv_curr = pd.read_csv(file_csv, header=None, names=["images", "masks"])
            # list of corresponding sample weights
            base_file_csv = utils.general.path_to_dataset(file_csv, self.cfg.dim)
            weight_curr = [self.cfg.weights_data[base_file_csv]]*len(csv_curr["images"])
            # add weights column
            weight_curr = pd.Series(weight_curr)
            csv_curr["weights"] = weight_curr.values
            csv_data = csv_data.append(csv_curr)

        image_filenames = csv_data["images"]
        weights = csv_data["weights"]
        mask_filenames = csv_data["masks"]

        dataset = tf.data.Dataset.from_tensor_slices((image_filenames, weights, mask_filenames))
        # apply map_fn on each sample
        dataset = dataset.map(self._map_fn, num_parallel_calls=8)
        # shuffle only when training; shuffle all examples
        if is_train:
           dataset = dataset.map(self._augmentation, num_parallel_calls=8)
           dataset = dataset.shuffle(len(csv_data["images"]), seed=self.cfg.seed)
        # set batch size
        dataset = dataset.batch(self.cfg.batch_size)
        # keep in memory
        dataset = dataset.prefetch(1)
        # loop forever
        dataset = dataset.repeat()
        # create iterator from dataset
        it = dataset.make_one_shot_iterator()
        X_op, w_op, y_op = it.get_next()
        return [X_op, w_op, y_op]

    def create_train_data_ops(self, files_csv):
        return self._create_data_ops(files_csv, True)

    def create_val_data_ops(self, files_csv):
        return self._create_data_ops(files_csv, False)

    def create_test_data_ops(self, files_csv):
        return self._create_data_ops(files_csv, False)

    def create_eval_data(self, files_csv):

        n_total, _ = utils.general.n_in_csv(files_csv)
        img_arr = np.zeros((n_total, self.cfg.height, self.cfg.width, 3), dtype=np.float32)
        mask_arr = np.zeros((n_total, self.cfg.height, self.cfg.width, 1), dtype=np.float32)
        name_arr = []

        ix = 0
        for file_csv in files_csv:

            df = pd.read_csv(file_csv, header=None)
            img_files, mask_files = df[0], df[1]

            for img_file, mask_file in zip(img_files, mask_files):
                img = cv2.imread(img_file)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
                img_arr[ix] = img/255.

                mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
                _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                mask = mask/255.
                mask_arr[ix] = np.expand_dims(mask, -1)
                name_arr.append(os.path.basename(img_file))

                ix += 1

        return img_arr, mask_arr, name_arr