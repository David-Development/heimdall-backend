import cv2
import dlib
import utils
import os


class FaceDetector:
    """
    Provides methods for face detection in images, specifically the Viola-Jones Method with haarcascades or lbpcascades
    from OpenCV  and the dlib face detector based on HOG.
    """

    def __init__(self, detection_method='dlib', model_path=None):
        """

        :param detection_method: Detection Method is either "opencv" or "dlib"
        :param model_path:  For "opencv" a path to the model file is needed
        """
        assert detection_method is 'opencv' or 'dlib', "Detection method must be either 'opencv' oder 'dlib'."

        if detection_method == 'opencv':
            assert model_path is not None, "For detection with OpenCV Haarcascades, a model file is needed!"

            if not os.path.exists(model_path):
                raise IOError("Model file for OpenCV not found!")

        self.detection_method = detection_method
        detector_method_name = 'create_' + detection_method + "_detector"
        detector_method = getattr(self, detector_method_name)

        self.model = detector_method(model_path)

    def detect(self, image):
        prediction_method_name = self.detection_method + "_predict"
        prediction_method = getattr(self, prediction_method_name)
        return prediction_method(image)

    def create_opencv_detector(self, model_path):
        return cv2.CascadeClassifier(model_path)

    def create_dlib_detector(self, model_path=None):
        return dlib.get_frontal_face_detector()

    def opencv_predict(self, image):
        return self.model.detectMultiScale(image, scaleFactor=1.01, minNeighbors=30, minSize=(80, 80),
                                           flags=cv2.CASCADE_SCALE_IMAGE)

    def dlib_predict(self, image):
        if image.shape[0] <= 150:
            return [face for face in self.model(image, 1)]
        else:
            return [face for face in self.model(image)]
