import os
from threading import Event

from flask import Flask
from flask_appconfig import AppConfig
from flask_restful import Api
from flask_socketio import SocketIO
from flask_sqlalchemy import SQLAlchemy

from celery import Celery

from recognition import Recognizer


basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)
api = Api(app)
AppConfig(app, os.path.join(basedir, 'default_config.py'))
db = SQLAlchemy(app)

# turn the flask app into a socketio app
socketio = SocketIO(app)

recognizer = Recognizer(shape_predictor_path=app.config['DLIB_SHAPE_PREDICTOR_PATH'],
                        descriptor_model_path=app.config['DLIB_FACE_RECOGNITION_MODEL_PATH'])

celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

from tasks import sync_db_from_filesystem

from camerathread import CameraThread
thread = CameraThread(app.config['CAMERA_SOCKET_HOST'], app.config['CAMERA_SOCKET_PORT'])
thread_stop_event = Event()

clf = tasks.create_classifier()
labels = {}
from app import models, resources

if app.config['SYNC_IMAGES_ON_START']:
    sync_db_from_filesystem()
