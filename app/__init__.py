import os

from flask import Flask
from flask_appconfig import AppConfig
from flask_restful import Api
from flask_sqlalchemy import SQLAlchemy

from celery import Celery

from recognition import Recognizer

basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)
api = Api(app)
AppConfig(app, os.path.join(basedir, 'default_config.py'))
db = SQLAlchemy(app)

recognizer = Recognizer(shape_predictor_path=app.config['DLIB_SHAPE_PREDICTOR_PATH'],
                        descriptor_model_path=app.config['DLIB_FACE_RECOGNITION_MODEL_PATH'])

celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

from tasks import sync_db_from_filesystem

clf = tasks.create_classifier()
from app import models, resources

if app.config['SYNC_IMAGES_ON_START']:
    sync_db_from_filesystem()
