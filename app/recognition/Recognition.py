import multiprocessing

from dlib import shape_predictor, face_recognition_model_v1
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

from . import detector


class Recognizer:
    """
    Face Recognizer based on Dlib's ResNet34 128-Dimensional feature vector
    """

    def __init__(self):
        self.shape_predictor = None
        self.recognition_model = None
        self.k = None
        self.X = None
        self.y = None
        self.clf_type = None
        self.clf_trained = None
        self.n_jobs = None
        self.clf = None

    def initialize(self, shape_predictor_path, descriptor_model_path, classifier="SVM", n_jobs=1, k=5):
        self.shape_predictor = shape_predictor(shape_predictor_path)
        self.recognition_model = face_recognition_model_v1(descriptor_model_path)
        self.k = k
        self.X = None
        self.y = None
        self.clf_type = classifier
        self.clf_trained = False

        if n_jobs == -1:
            self.n_jobs = multiprocessing.cpu_count()
        else:
            self.n_jobs = n_jobs

        if classifier is "kNN":
            self.clf = KNeighborsClassifier(k=self.k, n_jobs=n_jobs)
        else:
            self.clf = SVC(C=10, gamma=1, kernel='rbf', probability=True)
        pass

    def train_recognizer(self):
        """
        Trains the Classifier
        :return: 
        """
        self.check_data()
        transformed = []
        labels = []
        no_face = 0
        for data in zip(self.X, self.y):
            image = data[0]
            label = data[1]
            descriptors, _ = self.extract_descriptors(image)

            if len(descriptors) != 0:
                transformed.append(descriptors)
                labels.append(label)
            else:
                no_face += 1

        self.train_classifier(transformed, labels)

    def train_classifier(self, descriptors, labels):
        self.clf.fit(descriptors, labels)
        self.clf_trained = True

    def classify(self, image, dists=False):
        """
        :param image: 
        :param dists: 
        :return: 
        """

        if not self.clf_trained:
            descriptors, bbs = self.extract_descriptors(image)

            results = []
            for descriptor in descriptors:
                if self.clf_type is "kNN" and dists:
                    results.append(self.clf.kneighbors(self.k))
                else:
                    results.append(self.clf.predict_proba(descriptor))

            return results, bbs
        else:
            raise ValueError("Classifier must be trained before classifying!")

    def grid_search_and_apply(self, param_grid):
        """
        Perform gridsearch for internal classifier and apply best parameters
        :param param_grid: appropriate parameter grid for the internal classifier
        Example for SVM:
        kernels = ['linear', 'rbf']
        Cs = [0.001, 0.01, 0.1, 1, 10, 100]
        gammas = [0.001, 0.01, 0.1, 1, 10, 100]
        param_grid = {'C': Cs, 'gamma': gammas, 'kernel': kernels}
        :return: 
        """
        self.check_data()
        grid_cv = GridSearchCV(self.clf, param_grid=param_grid, n_jobs=self.n_jobs)
        grid_cv.fit(self.X, self.y)
        self.clf = grid_cv.best_estimator_

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

    def check_data(self):
        if self.X is None or self.y is None:
            raise ValueError('X and y must be set for this operation')

        if len(self.X) is not len(self.y):
            raise ValueError('X and y are not the same length, each image needs a single label')
