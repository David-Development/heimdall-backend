import eventlet
import os
import glob
import logging

from flask import Flask
from flask_appconfig import AppConfig
from flask_restful import Api
from flask_socketio import SocketIO
from flask_sqlalchemy import SQLAlchemy
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
r = redis.StrictRedis(host=app.config['REDIS_HOST'], port=app.config['REDIS_PORT'], db=1)

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
                          message_queue='redis://' + app.config['REDIS_HOST'] + ':' + app.config['REDIS_PORT'])
    else:
        r.flushdb()
        # Initialize SocketIO to emit events through the message queue.
        # Celery does not use eventlet. Therefore, we have to set async_mode
        # explicitly.
        socketio.init_app(None,
                          message_queue='redis://' + app.config['REDIS_HOST'] + ':' + app.config['REDIS_PORT'])
    celery.conf.update(flask_app.config)

    return None


def init_models():
    # Resync database on startup
    resources.resync_db()

    # Check for dlib models and download if necessary
    resources.check_models()

    from tasks import create_classifier, load_classifier

    db_model = models.ClassifierStats.query.filter_by(loaded=True).first()
    
    if db_model is None:
        app.clf = create_classifier()
        path = app.config['ML_MODEL_PATH'] + os.sep + '*.pkl'
        modelList = glob.glob(path)
        if modelList:
            latest_model = max(modelList, key=os.path.getctime)
            app.clf = load_classifier(latest_model)
            db_model = models.ClassifierStats.query.order_by(models.ClassifierStats.date.desc()).first()
            if db_model: # todo pre-compiled pkl files won't be recognized here on first startup...
                app.labels = db_model.labels_as_dict()
                db_model.loaded = True
            db.session.commit()
    else:
        app.clf = load_classifier(db_model.model_path)
        app.labels = db_model.labels_as_dict()
