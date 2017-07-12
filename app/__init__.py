# -*- coding: utf-8 -*-
import os

from flask import Flask
from flask_appconfig import AppConfig
from flask_restful import Api
from flask_sqlalchemy import SQLAlchemy


basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask('Heimdall')
api = Api(app)
AppConfig(app, os.path.join(basedir, 'default_config.py'))
db = SQLAlchemy(app)

from app import models, resources

from tasks import sync_db_from_filesystem

if app.config['SYNC_IMAGES_ON_START']:
    sync_db_from_filesystem()