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

    # Function default values are evaluated when the function is defined, not when it is called. Your default values
    # are set in stone the moment Python loads the module.
    # def __init__(self, begindate=datetime.datetime.now()):  # this does not work!

    def __init__(self, begindate=None):
        if begindate is None:
            begindate = datetime.datetime.now()

        self.begindate = begindate
