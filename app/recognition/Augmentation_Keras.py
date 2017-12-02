from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import logging
from time import time


class Augmenter:
    def __init__(self):
        self.generator = ImageDataGenerator(horizontal_flip=True, width_shift_range=0.05, rotation_range=10,
                                            height_shift_range=0.05, zoom_range=0.1)

        self.logger = logging.getLogger(__name__)

    def augment(self, images, labels, epochs, save_to_dir=None):
        self.logger.info("start augmentation of {} images, {} epochs.".format(len(labels), epochs))
        color = True
        images = np.asarray(images)
        # check for color images, shape is expected to be (n_images, width, height, n_channel)
        # if last dim is missing, its grayscale, for keras a dummy dimension is added
        print(images.shape)
        if images.ndim is 3:
            color = False
            images = np.asarray(images)[:, :, :, np.newaxis]

        start = time()
        x_data = []
        x_labels = []
        for e in range(epochs):
            d, l = self.generator.flow(images, labels, batch_size=len(labels), save_to_dir=save_to_dir).next()
            # if the images were grayscale in the beginning, remove the dummy dimension
            if not color:
                d = d[:, :, :, 0]

            x_data.extend(d.astype(np.uint8))
            x_labels.extend(l)

        # x_data = np.asarray(x_data)
        self.logger.info("end augmentation, generated {} images in {} seconds".format(len(x_labels), time() - start))
        return x_data, x_labels

    def augment_array_target(self, images, labels, target):
        color = True
        x_data = []
        x_labels = []

        if images.ndim is 3:
            images = images[:, :, :, np.newaxis]
            color = False

        #d, l = self.generator.flow(images, labels, batch_size=target).next()
        d, l = self.generator.flow(images, labels, batch_size=target, save_to_dir="./app/images/augmented_keras/").next()
        # remove dummy color channel
        if not color:
            d = d[:, :, :, 0]

        x_data.extend(d.astype(np.uint8))
        x_labels.extend(l)
        return x_data, x_labels

    def augment_array(self, images, labels, epochs, save_to_dir=None):
        color = True
        x_data = []
        x_labels = []

        for idx, img in enumerate(images):
            # keras wants color channels, even for grayscale images, so a dummy dimension in the end
            # (tensorflow format) is needed
            if img.ndim is 2:
                img = img[:, :, np.newaxis]
                color = False

            # keras wants an ndarray with shape [num_images, width, height, channels] so
            # for augmentation of single images a dummy dimension in the beginning is needed
            img = img[np.newaxis, :, :, :]
            for e in range(epochs):

                d, l = self.generator.flow(img, np.array([labels[idx]]), 1, save_to_dir=save_to_dir).next()

                # remove dummy color channel
                if not color:
                    d = d[:, :, :, 0]

                x_data.extend(d.astype(np.uint8))
                x_labels.extend(l)

        return x_data, x_labels
