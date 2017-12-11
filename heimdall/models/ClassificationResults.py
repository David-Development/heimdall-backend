# -*- coding: utf-8 -*-
from heimdall.app import db
from sqlalchemy.ext.hybrid import hybrid_property, hybrid_method

class ClassificationResults(db.Model):
    """
    Contains the classification results for the liveview
    """
    __tablename__ = 'classificationresults'

    id = db.Column(db.Integer, primary_key=True)
    clf_id = db.Column(db.Integer, db.ForeignKey('classifierstats.id', ondelete='CASCADE'))
    image_id = db.Column(db.Integer, db.ForeignKey('image.id'))
    image = db.relationship("Image")
    results = db.relationship('RecognitionResult', backref='classificationresults', lazy='dynamic')
    date = db.Column(db.DateTime)

    def __init__(self, clf_id, image_id, date):
        self.clf_id = clf_id
        self.image_id = image_id
        self.date = date

    @hybrid_property
    def num_persons(self):
        return self.results.count()