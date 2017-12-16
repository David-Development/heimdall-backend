# -*- coding: utf-8 -*-
import os


class Config(object):
    """Base configuration."""

    DOCKER = False

    BASEDIR = os.path.abspath(os.path.dirname(__file__)) # TODO rename to APP_DIR

    # MQTT Setup
    MQTT_CLIENT_ID = 'heimdall-backend'
    MQTT_BROKER_URL = 'mqtt-broker'
    MQTT_BROKER_PORT = 1883
    #MQTT_BROKER_PORT = 8083
    #MQTT_BROKER_TRANSPORT = 'websockets'
    MQTT_USERNAME = ''
    MQTT_PASSWORD = ''
    MQTT_REFRESH_TIME = 1.0  # refresh time in seconds
    MQTT_KEEPALIVE = 5
    MQTT_TLS_ENABLED = False

    # Postgres
    SQLALCHEMY_DATABASE_URI = 'postgresql://heimdall:heimdall@postgres/heimdall'
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Redis
    REDIS_HOST = 'redis'
    REDIS_PORT = '6379'

    IMAGE_FOLDER = 'images'
    NEW_IMAGES_FOLDER = os.path.join(IMAGE_FOLDER, 'new')
    SUBJECT_IMAGES_FOLDER = os.path.join(IMAGE_FOLDER, 'subjects')
    PLOTS_FOLDER = os.path.join(IMAGE_FOLDER, 'model_plots')
    UNKNOWN_IMAGES_FOLDER = os.path.join(SUBJECT_IMAGES_FOLDER, 'unknown')

    ML_MODEL_PATH = os.path.join(BASEDIR, 'ml_models')
    IMAGE_BASE_PATH = os.path.join(BASEDIR, IMAGE_FOLDER)
    PLOTS_BASE_PATH = os.path.join(BASEDIR, PLOTS_FOLDER)
    NEW_IMAGES_PATH = os.path.join(IMAGE_BASE_PATH, 'new')
    SUBJECTS_BASE_PATH = os.path.join(IMAGE_BASE_PATH, 'subjects')
    UNKNOWN_IMAGES_PATH = os.path.join(SUBJECTS_BASE_PATH, 'unknown')

    NUM_TARGET_IMAGES = 50
    PROBABILITY_THRESHOLD = 0.4

    DLIB_SHAPE_PREDICTOR_MODEL = 'shape_predictor_68_face_landmarks.dat'
    DLIB_FACE_RECOGNITION_MODEL = 'dlib_face_recognition_resnet_model_v1.dat'

    # Deprecated
    DLIB_SHAPE_PREDICTOR_MODEL_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
    DLIB_FACE_RECOGNITION_MODEL_URL = 'http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2'

    DLIB_SHAPE_PREDICTOR_PATH = os.path.join(ML_MODEL_PATH, DLIB_SHAPE_PREDICTOR_MODEL)
    DLIB_FACE_RECOGNITION_MODEL_PATH = os.path.join(ML_MODEL_PATH, DLIB_FACE_RECOGNITION_MODEL)

    WTF_CSRF_ENABLED = True
    PROPAGATE_EXCEPTIONS = True


class DockerConfig(Config):
    """Production configuration."""
    DOCKER = True


