# -*- coding: utf-8 -*-
import os

BASEDIR = os.path.abspath(os.path.dirname(__file__))

SERVER_NAME = '<ip/hostname>:<port>'

SQLALCHEMY_DATABASE_URI = 'postgresql://<db_user>:<password>@<host>/<db_name>'
SQLALCHEMY_TRACK_MODIFICATIONS = False

CELERY_BROKER_URL = 'redis://<host>:<port>/0'
CELERY_RESULT_BACKEND = 'redis://<host>:<port>/0'
CELERYD_MAX_TASKS_PER_CHILD = 1

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

DLIB_SHAPE_PREDICTOR_MODEL_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
DLIB_FACE_RECOGNITION_MODEL_URL = 'http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2'

DLIB_SHAPE_PREDICTOR_PATH = os.path.join(ML_MODEL_PATH, DLIB_SHAPE_PREDICTOR_MODEL)
DLIB_FACE_RECOGNITION_MODEL_PATH = os.path.join(ML_MODEL_PATH, DLIB_FACE_RECOGNITION_MODEL)

WTF_CSRF_ENABLED = True
PROPAGATE_EXCEPTIONS = True

