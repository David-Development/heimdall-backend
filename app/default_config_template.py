# -*- coding: utf-8 -*-
import os

BASEDIR = os.path.abspath(os.path.dirname(__file__))

# The secret key and recaptcha public key is automatically generated when the
# project is created
SECRET_KEY = 'pqeep35v-a1+-ul4!5)8^$3l5#mtszsk84*yxv-8p35%8e=ha'
RECAPTCHA_PUBLIC_KEY = '%r1(5##+iao!nn2=li*%mv)nn(q@gxw8-7n$fc4#_h&ea6h5k6-__'
SERVER_NAME = '<ip/hostname>:<port>'

SQLALCHEMY_DATABASE_URI = 'postgresql://<db_user>:<password>@<host>/<db_name>'
SQLALCHEMY_TRACK_MODIFICATIONS = False

CELERY_BROKER_URL = 'redis://<host>:<port>/0'
CELERY_RESULT_BACKEND = 'redis://<host>:<port>/0'

IMAGE_FOLDER = 'images'
NEW_IMAGES_FOLDER = os.path.join(IMAGE_FOLDER, 'new')
SUBJECT_IMAGES_FOLDER = os.path.join(IMAGE_FOLDER, 'subjects')
UNKNOWN_IMAGES_FOLDER = os.path.join(SUBJECT_IMAGES_FOLDER, 'unknown')

ML_MODEL_PATH = os.path.join(BASEDIR, 'ml_models')
IMAGE_BASE_PATH = os.path.join(BASEDIR, IMAGE_FOLDER)
NEW_IMAGES_PATH = os.path.join(IMAGE_BASE_PATH, 'new')
SUBJECTS_BASE_PATH = os.path.join(IMAGE_BASE_PATH, 'subjects')
UNKNOWN_IMAGES_PATH = os.path.join(SUBJECTS_BASE_PATH, 'unknown')

DLIB_SHAPE_PREDICTOR_MODEL = 'shape_predictor_68_face_landmarks.dat'
DLIB_FACE_RECOGNITION_MODEL = 'dlib_face_recognition_resnet_model_v1.dat'

DLIB_SHAPE_PREDICTOR_MODEL_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
DLIB_FACE_RECOGNITION_MODEL_URL = 'http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2'

DLIB_SHAPE_PREDICTOR_PATH = os.path.join(ML_MODEL_PATH, DLIB_SHAPE_PREDICTOR_MODEL)
DLIB_FACE_RECOGNITION_MODEL_PATH = os.path.join(ML_MODEL_PATH, DLIB_FACE_RECOGNITION_MODEL)

WTF_CSRF_ENABLED = True
PROPAGATE_EXCEPTIONS = True

SYNC_IMAGES_ON_START = False
