from dlib import shape_predictor, face_recognition_model_v1
from . import detector


class Recognizer:
    def __init__(self, shape_predictor_path, descriptor_model_path):
        self.shape_predictor = shape_predictor(shape_predictor_path)
        self.recognition_model = face_recognition_model_v1(descriptor_model_path)
        self.initialized = True

    def extract_descriptors(self, image):
        """
        Detect faces, extract descriptors and return the descriptor and the bounding box of the found faces
        :param image: 
        :return: descriptors and corresponding bounding boxes
        """
        faces = detector.detect(image)
        descriptors = []
        for face in faces:
            shape = self.shape_predictor(image, face)
            descriptors.append(self.recognition_model.compute_face_descriptor(image, shape))
        return descriptors, faces
