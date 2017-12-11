# -*- coding: utf-8 -*-
import os
import datetime
from heimdall.app import db
from sqlalchemy.ext.hybrid import hybrid_property, hybrid_method

class ClassifierStats(db.Model):
    """
    Represents a trained model and its attributes
    """
    __tablename__ = 'classifierstats'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255))
    # SVM or kNN
    classifier_type = db.Column(db.String(10))
    # path to model file
    model_path = db.Column(db.String())
    # date of training
    date = db.Column(db.DateTime)
    # score of cross validation
    cv_score = db.Column(db.Float)
    # corresponding labels for classification results
    labels = db.relationship('Labels', backref='classifierstats', lazy='dynamic')
    # number of classes/persons
    num_classes = db.Column(db.Integer)
    # total number of images used for training
    total_images = db.Column(db.Integer)
    # average number of images per class
    avg_base_img = db.Column(db.Float)
    # total number of images with no faces found (including augmented images)
    total_no_face = db.Column(db.Integer)
    # total training time, consisting of time for augmentation, feature extraction and classifier training, in seconds
    training_time = db.Column(db.Integer)
    # is model currently loaded/used
    loaded = db.Column(db.Boolean)
    # path to confusion plot
    confusion_matrix = db.Column(db.String())
    # path to learning curve plot
    learning_curve = db.Column(db.String())

    def __init__(self, name, classifier_type, model_path, date, num_classes, cv_score=None, total_images=None,
                 avg_base_img=None, total_no_face=None, training_time=None, confusion_matrix=None, learning_curve=None):
        self.name = name
        self.classifier_type = classifier_type
        self.model_path = model_path
        self.date = date
        self.num_classes = num_classes
        self.cv_score = cv_score
        self.total_images = total_images
        self.avg_base_img = avg_base_img
        self.total_no_face = total_no_face
        self.training_time = training_time
        self.confusion_matrix = confusion_matrix
        self.learning_curve = learning_curve

    @hybrid_method
    def labels_as_dict(self):
        """
        Creates a dictionary for the number labels of the trained model to person names.
        :return: dictionary for numbers to labels
        """
        label_dict = {}
        for label in self.labels:
            label_dict[label.num] = label.label

        return label_dict

    @hybrid_property
    def confusion_url(self):
        """
        url to the confusion matrix image
        :return:
        """
        path_components = self.confusion_matrix.split(os.sep)
        starting_index = path_components.index('images')
        return '/'.join(path_components[starting_index:])

    @hybrid_property
    def learning_curve_url(self):
        """
        url to the learning curve image
        :return:
        """
        path_components = self.learning_curve.split(os.sep)
        starting_index = path_components.index('images')
        return '/'.join(path_components[starting_index:])
