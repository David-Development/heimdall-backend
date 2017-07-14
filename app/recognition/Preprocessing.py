from collections import defaultdict

import FaceDetection
import FaceAligment
import cv2
from PIL import Image
import numpy as np
import utils
import logging
import openface


class Preprocessor:
    """
    Class for Preprocessing of images. Consists of detecting faces and aligning them. Histogram Equalization can also be
    performed.
    """

    def __init__(self, detection_method="opencv", detection_model_path=None, alignment=True, prediction_model_path=None,
                 size=None, heq=True, clahe=True, use_original=False):
        """
        :param detection_method: Method of Face Detection, either "opencv" or "dlib"
        :param detection_model_path: If Face Detection is "opencv" a path to the model file is needed
        :param alignment:  True or False, depending if detected faces should be aligned
        :param prediction_model_path: If detected faces should be aligned, a path to the dlib predictor file is needed
        :param size: The width to which the image should be scaled, keeps the ratio.
        :param heq: True or False, depending if Histogram Equalization should be performed.
        :param clahe: True or False, depending if contrast limiting adaptive histogram equalization should be used
        """
        self.logger = logging.getLogger(__name__)

        self.alignment = alignment
        self.heq = heq
        self.clahe = clahe
        self.size = size
        self.detection_method = detection_method

        if detection_method == "opencv" and detection_model_path is None:
            self.logger.info("opencv needs a model file, falling back to dlib detector")
            self.detection_method = "dlib"

        self.detector = FaceDetection.FaceDetector(self.detection_method, detection_model_path)
        self.logger.info("Detector initialized with detection method: " + self.detection_method)

        if alignment:
            assert prediction_model_path is not None
            if use_original:
                self.aligner = openface.AlignDlib(prediction_model_path)
            else:
                self.aligner = FaceAligment.FaceAlignment(prediction_model_path)

            self.logger.info("Landmark Predictor initialized")

    def preprocess(self, image, bounding_box=False):
        """
        Preprocesses the image with the methods defined for the class instance

        :param image: the image to preprocess
        :param bounding_box: boolean, if True, returns the original image with the corresponding bounding box. In this
        case, HEQ ist not performed.
        :return: image or bounding box
        """
        self.logger.debug("starting preprocessing")
        processed_faces = []
        if image.ndim != 2:
            color = True
        else:
            color = False

        # opencv needs a grayscale image...
        if self.detection_method == "opencv" and color:
            gray = np.asarray(Image.fromarray(image).convert('L')).copy()
            faces = self.detect(gray)
        else:
            faces = self.detector.detect(image)

        if len(faces) == 0:
            return None, None

        self.logger.debug("found " + str(len(faces)) + " faces")

        ##dlib bb is too small

        if self.detection_method is "dlib" and not bounding_box:
            self.logger.debug("Resizing BoundingBox")
            faces = [enlarge_dlib_rectangle(image, face, 0.8) for face in faces]

        if bounding_box:
            return image, utils.cv2dlib(faces[0])

        if self.alignment:
            self.logger.debug("starting alignment")
            for face in faces:
                face = utils.cv2dlib(face)
                processed_faces.append(self.aligner.align(rgbImg=image, imgDim=self.size, bb=face))
        else:
            self.logger.debug("no alignment, just cropping and resizing")
            processed_faces = [cv2.resize(utils.croprect(image, face), (self.size, self.size))
                               for face in faces]

        if self.heq:
            if color:
                self.logger.warn("Histogram Equalization not yet implemented for color images, skipping...")
            else:
                if self.clahe:
                    self.logger.debug("starting histogram equalization using clahe")
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(20, 20))
                    processed_faces = [clahe.apply(face) for face in processed_faces]
                else:
                    self.logger.debug("starting histogram equalization")
                    processed_faces = [cv2.equalizeHist(face) for face in processed_faces]

        # Adding grayscale axis for keras/tf
        # processed_faces = [face[..., np.newaxis] for face in processed_faces]
        self.logger.debug("preprocessing done")
        return processed_faces, faces

    def preprocess_all_bbs(self, images, labels):
        """
        Same as preprocess_all, but instead of returning the cropped images, it just returns the original images and
        the corresponding bounding boxes in dlib format, useful for the dlib embedding. HEQ ist not performed.
        :param images:
        :param labels:
        :return:
        """

        faces = []
        bbs = []
        processed_labels = []
        no_face_detected = {}

        num_images = len(images)
        self.logger.info("start batch bbs preprocessing of " + str(num_images) + " images")
        for idx in range(num_images):
            face, bb = self.preprocess(images[idx], bounding_box=True)
            if face is not None:
                faces.append(face)
                bbs.append(bb)
                processed_labels.append(labels[idx])
            else:
                self.logger.info("no face found, subject: " + str(labels[idx]))
                no_face_detected[labels[idx]] = images[idx]

        self.logger.info("batch preprocessing with bounding boxes done")
        return faces, processed_labels, bbs, no_face_detected

    def preprocess_all(self, images, labels):
        processed_faces = []
        processed_labels = []
        no_face_detected = defaultdict(list)

        # imgs = np.asarray(images)
        num_images = len(images)

        self.logger.info("start batch preprocessing of " + str(num_images) + " images")
        for idx in range(num_images):
            face, _ = self.preprocess(images[idx])
            # Only first found face
            if face is not None:
                processed_faces.append(face[0])
                processed_labels.append(labels[idx])
            else:
                self.logger.info("no face found, subject: " + str(labels[idx]))
                no_face_detected[labels[idx]].append(images[idx])

        self.logger.info("batch preprocessing done")
        return processed_faces, processed_labels, no_face_detected

    def detect(self, image):
        return self.detector.detect(image)


def enlarge_dlib_rectangle(img, rect, p):
    x, y, w, h = [v for v in rect]

    x -= int((w * p) / 2)
    y -= int((h * p) / 2)
    w = int(w + w * p)
    h = int(h + h * (p))

    if x < 0:
        x = 0
    if y < 0:
        y = 0
    if w > img.shape[0]:
        w = img.shape[0] - y
    if h > img.shape[1]:
        h = img.shape[1] - x
    return [x, y, w, h]
