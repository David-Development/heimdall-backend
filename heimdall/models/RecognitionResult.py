# -*- coding: utf-8 -*-
from heimdall.app import db
from sqlalchemy.ext.hybrid import hybrid_property, hybrid_method


class RecognitionResult(db.Model):
    """
    Represents a classification result from a single face inside an image.
    """
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
        """
        Converts a list with bounding box coordinates to single values for db storage and back
        :return:
        """
        return [self.x, self.y, self.w, self.h]
