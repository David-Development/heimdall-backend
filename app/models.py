# -*- coding: utf-8 -*-
import os
import datetime
from app import db
from sqlalchemy.ext.hybrid import hybrid_property, hybrid_method


class Gallery(db.Model):
    __tablename__ = 'gallery'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), unique=True)
    path = db.Column(db.String())
    subject_gallery = db.Column(db.Boolean, default=True)
    images = db.relationship('Image', backref='gallery', lazy='dynamic')
    createdate = db.Column(db.DateTime)

    def __init__(self, name, path, subject_gallery=True):
        self.name = name
        self.path = path
        self.subject_gallery = subject_gallery
        self.createdate = datetime.datetime.now()

    @hybrid_property
    def images_count(self):
        return self.images.count()

    def __repr__(self):
        return '<Gallery %r>' % self.name


class Image(db.Model):
    __tablename__ = 'image'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255))
    gallery_id = db.Column(db.Integer, db.ForeignKey('gallery.id'))
    path = db.Column(db.String())
    createdate = db.Column(db.DateTime)

    @property
    def url(self):
        path_components = self.path.split(os.sep)
        return '/'.join(path_components)

    def __init__(self, name, path, gallery_id=None):
        self.name = name
        self.path = path
        self.gallery_id = gallery_id
        self.createdate = datetime.datetime.now()


class ClassifierStats(db.Model):
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

    def __init__(self, name, classifier_type, model_path, date, num_classes, cv_score=None, total_images=None,
                 avg_base_img=None,
                 total_no_face=None, training_time=None):
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

    @hybrid_method
    def labels_as_dict(self):
        label_dict = {}
        for label in self.labels:
            label_dict[label.num] = label.label

        return label_dict


class Labels(db.Model):
    __tablename__ = 'labels'

    id = db.Column(db.Integer, primary_key=True)
    clf_id = db.Column(db.Integer, db.ForeignKey('classifierstats.id'))
    num = db.Column(db.Integer)
    label = db.Column(db.String(255))


class ClassificationResults(db.Model):
    """
    Contains the classification results for the liveview
    """
    __tablename__ = 'classificationresults'

    id = db.Column(db.Integer, primary_key=True)
    clf_id = db.Column(db.Integer, db.ForeignKey('classifierstats.id', ondelete='CASCADE'))
    image_id = db.Column(db.Integer, db.ForeignKey('image.id'))
    image = db.relationship("Image")
    results = db.relationship('Result', backref='classificationresults', lazy='dynamic')
    date = db.Column(db.DateTime)

    def __init__(self, clf_id, image_id, date):
        self.clf_id = clf_id
        self.image_id = image_id
        self.date = date

    @hybrid_property
    def num_persons(self):
        return self.results.count()


class Result(db.Model):
    __tablename__ = 'result'

    id = db.Column(db.Integer, primary_key=True)
    classification = db.Column(db.Integer, db.ForeignKey('classificationresults.id', ondelete='CASCADE'))
    gallery_id = db.Column(db.Integer, db.ForeignKey('gallery.id'))
    gallery = db.relationship('Gallery')
    probability = db.Column(db.Float)
    x = db.Column(db.Integer)
    y = db.Column(db.Integer)
    w = db.Column(db.Integer)
    h = db.Column(db.Integer)

    def __init__(self, classification, gallery_id, probability, bounding_box=None):
        self.classification = classification
        self.gallery_id = gallery_id
        self.probability = probability
        if bounding_box is not None:
            self.x = bounding_box[0]
            self.y = bounding_box[1]
            self.w = bounding_box[2]
            self.h = bounding_box[3]

    @hybrid_property
    def bounding_box(self):
        return [self.x, self.y, self.w, self.h]
