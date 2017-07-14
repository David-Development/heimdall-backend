import os

from flask import Flask
from flask_appconfig import AppConfig
from flask_restful import Api
from flask_sqlalchemy import SQLAlchemy

from celery import Celery

from recognition import Recognizer

basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask('Heimdall')
api = Api(app)
AppConfig(app, os.path.join(basedir, 'default_config.py'))
db = SQLAlchemy(app)

recognizer = Recognizer()

celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

from app import models, resources

from tasks import sync_db_from_filesystem

if app.config['SYNC_IMAGES_ON_START']:
    sync_db_from_filesystem()
