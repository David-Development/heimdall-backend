# -*- coding: utf-8 -*-
import datetime
from heimdall.app import db
from sqlalchemy.ext.hybrid import hybrid_property, hybrid_method


class Gallery(db.Model):
    """
    Represents a gallery of a person.
    """
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