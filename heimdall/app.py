import glob
import os

import redis as red
import eventlet
from flask import Flask
from flask_mqtt.flask_mqtt import Mqtt
from flask_restful import Api
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS

from heimdall.settings import DockerConfig

#eventlet.monkey_patch()


clf = None   # TODO check if this can be deleted
labels = {}  # TODO check if this can be deleted

# This import needs to come after we initialized the database

db = None
mqtt = None
api = None
redis = None
recognizer = None
app = None


def create_app(config_object=DockerConfig):
    global db
    global mqtt
    global api
    global redis
    global recognizer
    global app

    app = Flask(__name__)
    app.config.from_object(config_object)
    #print(app.config)

    CORS(app)


    # Init API
    api = Api(app)

    # Init MQTT
    mqtt = Mqtt(app)
    mqtt.subscribe('camera')

    # Init DB
    db = SQLAlchemy(app)

    # Init Redis
    redis = red.StrictRedis(host=app.config['REDIS_HOST'],
                              port=app.config['REDIS_PORT'],
                              db=1,
                              charset="utf-8",
                              decode_responses=True)

    # Init Recognizer
    from heimdall.recognition.Recognition import Recognizer
    recognizer = Recognizer(
                shape_predictor_path=app.config['DLIB_SHAPE_PREDICTOR_PATH'],
                descriptor_model_path=app.config['DLIB_FACE_RECOGNITION_MODEL_PATH'])

    from heimdall.recognition import RecognitionManager
    RecognitionManager.init(app, db)

    return app

