# -*- coding: utf-8 -*-
import os
from app import db
from sqlalchemy.ext.hybrid import hybrid_property, hybrid_method


class Gallery(db.Model):
    __tablename__ = 'gallery'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), unique=True)
    path = db.Column(db.String())
    subject_gallery = db.Column(db.Boolean, default=True)
    images = db.relationship('Image', backref='gallery', lazy='dynamic')

    def __init__(self, name, path, subject_gallery=True):
        self.name = name
        self.path = path
        self.subject_gallery = subject_gallery

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

    @property
    def url(self):
        path_components = self.path.split(os.sep)
        return '/'.join(path_components)

    def __init__(self, name, path, gallery_id):
        self.name = name
        self.path = path
        self.gallery_id = gallery_id


class ClassifierStats(db.Model):
    __tablename__ = 'classifierstats'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255))
    classifier_type = db.Column(db.String(10))
    model_path = db.Column(db.String())
    date = db.Column(db.DateTime)
    cv_score = db.Column(db.Float)
    labels = db.relationship('Labels', backref='classifierstats', lazy='dynamic')

    def __init__(self, name, classifier_type, model_path, date, cv_score=None):
        self.name = name
        self.classifier_type = classifier_type
        self.model_path = model_path
        self.date = date
        self.cv_score = cv_score

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
