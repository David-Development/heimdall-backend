# -*- coding: utf-8 -*-
from heimdall.app import db

class Labels(db.Model):
    """
    Represents a association from a number to a certain label for a trained model
    """
    __tablename__ = 'labels'

    id = db.Column(db.Integer, primary_key=True)
    clf_id = db.Column(db.Integer, db.ForeignKey('classifierstats.id'))
    num = db.Column(db.Integer)
    label = db.Column(db.String(255))
