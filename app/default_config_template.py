# -*- coding: utf-8 -*-
import os
BASEDIR = os.path.abspath(os.path.dirname(__file__))

# The secret key and recaptcha public key is automatically generated when the
# project is created
SECRET_KEY = 'pqeep35v-a1+-ul4!5)8^$3l5#mtszsk84*yxv-8p35%8e=ha'
RECAPTCHA_PUBLIC_KEY = '%r1(5##+iao!nn2=li*%mv)nn(q@gxw8-7n$fc4#_h&ea6h5k6-__'

SQLALCHEMY_DATABASE_URI = 'postgresql://<db_user>:<password>@<host>/<db_name>'
SQLALCHEMY_TRACK_MODIFICATIONS = False

IMAGE_FOLDER = 'images'
NEW_IMAGES_FOLDER = os.path.join(IMAGE_FOLDER, 'new')
UNKNOWN_IMAGES_FOLDER = os.path.join(IMAGE_FOLDER, 'unknown')
SUBJECT_IMAGES_FOLDER = os.path.join(IMAGE_FOLDER, 'subjects')

ML_MODEL_PATH = os.path.join(BASEDIR, 'ml_models')
IMAGE_BASE_PATH = os.path.join(BASEDIR, IMAGE_FOLDER)
NEW_IMAGES_PATH = os.path.join(IMAGE_BASE_PATH, 'new')
UNKNOWN_IMAGES_PATH = os.path.join(IMAGE_BASE_PATH, 'unknown')
SUBJECTS_BASE_PATH = os.path.join(IMAGE_BASE_PATH, 'subjects')

WTF_CSRF_ENABLED = True
PROPAGATE_EXCEPTIONS = True

SYNC_IMAGES_ON_START = False
