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
        return self.path.replace(os.sep, '/')

    def __init__(self, name, path, gallery_id):
        self.name = name,
        self.path = path,
        self.gallery_id = gallery_id
