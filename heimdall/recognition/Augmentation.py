import Augmentor
import numpy as np
import logging
import cv2
import os
import shutil
from . import utils


class Augmenter:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def augment_array_target(self, subject_name, batch_size):

        subject_path = "./heimdall/images/subjects/" + subject_name + "/"
        pipeline = Augmentor.Pipeline(subject_path)
        pipeline.rotate(probability=0.5, max_left_rotation=10, max_right_rotation=10)
        pipeline.zoom(probability=0.5, min_factor=1.0, max_factor=1.1)
        pipeline.sample(batch_size)

        x_data = []

        subject_path_augmented = subject_path + "/output/"
        for filename in os.listdir(subject_path_augmented):
            file_path = os.path.join(subject_path_augmented, filename)
            if filename.endswith('.JPEG'):
                #self.logger.info("Reading file: %s", filename)
                im = utils.load_image(file_path)
                im = np.asarray(im, dtype=np.uint8)
                im.setflags(write=True)
                x_data.append(im)

        shutil.rmtree(subject_path_augmented)
        return x_data
