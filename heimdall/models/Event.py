# -*- coding: utf-8 -*-
import os
import datetime
from heimdall.app import db

class Event(db.Model):
    """
    Represents an event
    """
    __tablename__ = 'event'

    id = db.Column(db.Integer, primary_key=True)
    begindate = db.Column(db.DateTime)

    def __init__(self, begindate=datetime.datetime.now()):
        self.begindate = begindate
