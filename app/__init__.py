import eventlet
import os
import glob
import logging

from flask import Flask
from flask_appconfig import AppConfig
from flask_restful import Api
from flask_socketio import SocketIO
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from celery import Celery
import redis

from recognition import Recognizer

logging.getLogger('socketio').setLevel(logging.ERROR)
logging.getLogger('engineio').setLevel(logging.ERROR)

eventlet.monkey_patch()

basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)
api = Api(app)
AppConfig(app, os.path.join(basedir, 'default_config.py'))
db = SQLAlchemy(app)
r = redis.StrictRedis(host='localhost', port='6379', db=1)

# turn the flask app into a socketio app

socketio = SocketIO()

recognizer = Recognizer(shape_predictor_path=app.config['DLIB_SHAPE_PREDICTOR_PATH'],
                        descriptor_model_path=app.config['DLIB_FACE_RECOGNITION_MODEL_PATH'])

celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

clf = None
labels = {}
from app import models, resources


def create_app(main=True):
    # Initialize extensions
    extensions(app, main)

    return app


def extensions(flask_app, main):
    logger = False

    if main:
        # Initialize SocketIO as server and connect it to the message queue.
        socketio.init_app(flask_app,
                          message_queue='redis://')
    else:
        # Initialize SocketIO to emit events through the message queue.
        # Celery does not use eventlet. Therefore, we have to set async_mode
        # explicitly.
        socketio.init_app(None,
                          message_queue='redis://')
    celery.conf.update(flask_app.config)

    return None


def init_models():
    from tasks import create_classifier, load_classifier
    app.clf = create_classifier()
    path = app.config['ML_MODEL_PATH'] + os.sep + '*.pkl'
    latest_model = max(glob.glob(path), key=os.path.getctime)
    app.clf = load_classifier(latest_model)
    db_model = models.ClassifierStats.query.order_by(models.ClassifierStats.date.desc()).first()
    app.labels = db_model.labels_as_dict()
