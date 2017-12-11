# -*- coding: utf-8 -*-
import os
import datetime
from heimdall.app import db

class Image(db.Model):
    """
    Represents an image.
    """
    __tablename__ = 'image'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255))
    gallery_id = db.Column(db.Integer, db.ForeignKey('gallery.id'))
    path = db.Column(db.String())
    createdate = db.Column(db.DateTime)

    @property
    def url(self):
        """
        create an url for this image
        """
        path_components = self.path.split(os.sep)
        return '/'.join(path_components)

    def __init__(self, name, path, gallery_id=None):
        self.name = name
        self.path = path
        self.gallery_id = gallery_id
        self.createdate = datetime.datetime.now()