import os
import glob
import datetime
import json

import binascii
from flask_restful import Resource, reqparse, marshal_with, fields, abort
from sqlalchemy.exc import IntegrityError
from flask import jsonify, send_from_directory, request, url_for, render_template, json, Response, abort
from io import BytesIO
from PIL import Image as PILImage
import numpy as np
import requests
from heimdall.models.Gallery import Gallery
from heimdall.models.Image import Image
from heimdall.models.Event import Event
from heimdall.models.ClassifierStats import ClassifierStats
from heimdall.models.ClassificationResults import ClassificationResults
from heimdall.models.RecognitionResult import RecognitionResult

from heimdall.app import api, app, db, recognizer, clf, labels, redis, mqtt
from heimdall.recognition import utils
from heimdall.recognition.RecognitionManager import recognition_manager
from heimdall.tasks import (sync_db_from_filesystem, delete_gallery, move_images, download_models, models_exist,
                            TrainRecognizer, load_classifier, new_image, clear_gallery)

config = app.config

gallery_parser = reqparse.RequestParser()
gallery_parser.add_argument('name')

image_parser = reqparse.RequestParser()
image_parser.add_argument('image_ids', action="append")
image_parser.add_argument('gallery_id')

live_parser = reqparse.RequestParser()
live_parser.add_argument('image')
live_parser.add_argument('annotate')

image_upload_parser = reqparse.RequestParser()
image_upload_parser.add_argument('image')

train_recognizer = TrainRecognizer()


@mqtt.on_message()
def handle_mqtt_message(client, userdata, message):
    print("mqtt.on_message():", message.topic)


@mqtt.on_topic('camera')
def handle_mytopic(client, userdata, message):
    topic = message.topic
    print("===========")
    print("Topic:", topic)

    if topic == "camera":
        try:
            image = message.payload.decode()
            filename = str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')[:-3]) + '.jpg'
            id = new_image(image, filename)
            recognition_manager.add_image(image_id=id)
        except binascii.Error as err:
            print(err)



'''
@mqtt.on_log()
def handle_logging(client, userdata, level, buf):
    print(level, buf)
'''



gallery_fields = {
    'id': fields.Integer,
    'name': fields.String,
    'path': fields.String,
    'images': fields.Integer(attribute='images_count'),
    'subject_gallery': fields.Boolean,
    'date': fields.DateTime(attribute='createdate'),
}

image_fields = {
    'id': fields.Integer,
    'name': fields.String,
    'path': fields.String,
    'url': fields.String,
    'gallery_id': fields.Integer(attribute='gallery.id'),
    'gallery_name': fields.String(attribute='gallery.name'),
    'date': fields.DateTime(attribute='createdate'),
}
label_fields = {
    'id': fields.Integer,
    'clf_id': fields.Integer,
    'num': fields.Integer,
    'label': fields.String,
}

model_fields = {
    'id': fields.Integer,
    'name': fields.String,
    'classifier_type': fields.String,
    'date': fields.DateTime,
    'cv_score': fields.Float,
    'avg_base_img': fields.Float,
    'total_images': fields.Integer,
    'total_no_faces': fields.Integer,
    'loaded': fields.Boolean,
    'num_classes': fields.Integer,
    'training_time': fields.Integer,
    'confusion_url': fields.String,
    'learning_curve_url': fields.String,
    'labels': fields.List(fields.Nested(label_fields))
}

model_list_fields = {
    'data': fields.List(fields.Nested(model_fields))
}


class GalleryRes(Resource):
    @marshal_with(gallery_fields)
    def get(self, gallery_id):
        gallery = Gallery.query.filter_by(id=gallery_id).first()
        if gallery is None:
            abort(409, descripction="Gallery not found")
        return gallery

    @marshal_with(gallery_fields)
    def post(self):
        parsed_args = gallery_parser.parse_args()
        name = parsed_args['name']
        gallery = Gallery(name=name, path=os.path.join(config['SUBJECT_IMAGES_FOLDER'], name))

        # prevent empty name
        if name is None or name is '':
            abort(409, description="A gallery must have a name!")

        try:
            db.session.add(gallery)
            db.session.commit()
        except IntegrityError:
            abort(409, description="A gallery with that name already exists.")

        os.mkdir(os.path.join(config['BASEDIR'], gallery.path))
        return gallery, 201

    def delete(self, gallery_id):
        # User wants to delete a gallery, images get sent back to 'new' gallery
        # gallery folder gets deleted
        gallery_new = Gallery.query.filter_by(name="new").first()
        gallery_old = Gallery.query.filter_by(id=gallery_id).first()

        delete_gallery(gallery_old, gallery_new)
        images = Image.query.filter_by(gallery_id=gallery_id).all()

        for image in images:
            image.gallery_id = gallery_new.id
            image.path = os.path.join(gallery_new.path, image.name)

        # Delete all Results for this Person/Gallery
        RecognitionResult.query.filter_by(gallery_id=gallery_id).delete()
        Gallery.query.filter_by(id=gallery_id).delete()
        db.session.commit()

        return {'message': 'gallery deleted'}, 200


class GalleryListRes(Resource):
    @marshal_with(gallery_fields, envelope='galleries')
    def get(self):
        galleries = Gallery.query.all()
        return galleries


class ImageListRes(Resource):
    @marshal_with(image_fields)
    def get(self):
        return Image.query.all()

    def put(self):
        """
        Move Images from one Gallery to another
        :return: 
        """
        parsed_args = image_parser.parse_args()
        gallery_id = parsed_args['gallery_id']
        image_ids = parsed_args['image_ids']

        gallery = Gallery.query.filter_by(id=gallery_id).first()
        images = Image.query.filter(Image.id.in_(image_ids)).all()

        # move on filesystem
        move_images(gallery, images)
        # move in db
        for image in images:
            image.gallery_id = gallery_id
            image.path = os.path.join(gallery.path, image.name)
            ClassificationResults.query.filter_by(image_id=image.id).delete()

        db.session.commit()

        return {'message': 'images moved'}, 200


class GalleryImagesListRes(Resource):
    @marshal_with(image_fields)
    def get(self, gallery_id):
        images = Image.query.order_by(Image.id.desc()).filter_by(gallery_id=gallery_id).all()
        return images


class ModelListRes(Resource):
    @marshal_with(model_list_fields)
    def get(self):
        res = ClassifierStats.query.order_by(ClassifierStats.loaded.desc(), ClassifierStats.date.desc()).all()
        return {'data': res}, 200


class ModelRes(Resource):
    @marshal_with(model_fields)
    def get(self, classifier_id):
        res = ClassifierStats.query.filter_by(id=classifier_id).first()
        return res, 200


api.add_resource(GalleryRes, '/api/gallery/', '/api/gallery/<gallery_id>/')
api.add_resource(GalleryImagesListRes, '/api/gallery/<gallery_id>/images/')
api.add_resource(GalleryListRes, '/api/galleries/')
api.add_resource(ImageListRes, '/api/images/')
api.add_resource(ModelListRes, '/api/classifier/')
api.add_resource(ModelRes, '/api/classifier/<classifier_id>/')


@app.route("/api/gallery/<gallery_id>/clear", methods=['POST'])
def clear_selected_gallery(gallery_id):
    gallery = Gallery.query.filter_by(id=gallery_id).first()
    if gallery is None:
        abort(409, description="Gallery not found")

    clear_gallery(gallery)

    return jsonify({'message': 'gallery cleared'})


@app.route("/")
def index():
    return render_template('index.html', title="Heimdall Face Recognition")


@app.route("/liveview")
def liveview():
    return render_template('liveview.html', title="Liveview")


@app.route("/galleries")
def galleries():
    gls = Gallery.query.order_by(Gallery.id.asc()).all()
    return render_template('galleries.html', title="Person Galleries", galleries=gls)


@app.route("/classifications")
def recent_classifications():
    classifications = ClassificationResults.query.order_by(ClassificationResults.date.desc()).limit(50).all()
    gls = Gallery.query.all()
    return render_template('recent_classifications.html', title="Recent Classifications",
                           classifications=classifications, galleries=gls)


def getDetectionResults(image_id):
    detectionList = []
    for classif in ClassificationResults.query.filter(ClassificationResults.image_id == image_id).all():
        res = classif.results.first()
        empDict = {
            #'id': classif.id,
            'id': res.gallery_id,
            'location': [res.x, res.y, res.w, res.h]
        }
        detectionList.append(empDict)
    #print(detectionList)
    return detectionList

def getImagesForEvent(event_id):
    imageList = []
    for image in Image.query.filter(Image.event_id == event_id).limit(5).all():
        empDict = {
            'id': image.id,
            'url': "http://localhost:5000/" + image.path,
            'detected': getDetectionResults(image.id)
        }
        imageList.append(empDict)
    return imageList

@app.route("/events/", methods=['GET'])
def getEvents():
    eventList = []
    for event in Event.query.order_by(Event.begindate).all():
        empDict = {
            'id': event.id,
            'date': event.begindate.date().isoformat(),
            'time': event.begindate.time().isoformat(),
            'images': getImagesForEvent(event.id)
        }

        if len(empDict['images']) > 0:
            eventList.append(empDict)

    return jsonify(eventList), 201

@app.route("/persons/", methods=['GET'])
def getPersons():
    galleryList = []
    for gallery in Gallery.query.filter(Gallery.name != "unkown").all():
        empDict = {
            'id': gallery.id,
            'name': gallery.name,
            'path': "http://localhost:5000/" + gallery.path,
            'subject_gallery': gallery.subject_gallery,
            'images': gallery.images.count(),
            'avatar': gallery.images.first().path
        }

        galleryList.append(empDict)

    return jsonify(galleryList), 201


def getImagesForPerson(query):
    imageList = []
    for image in query.all():
        empDict = {
            'id': image.id,
            'path': "http://localhost:5000/" + image.path
        }
        imageList.append(empDict)
    return imageList

@app.route("/person/<int:person_id>/", methods=['GET'])
def getPersonById(person_id):
    empDict = {}
    gallery = Gallery.query.filter(Gallery.id == person_id).first()
    if gallery:
        empDict = {
            'id': gallery.id,
            'name': gallery.name,
            'path': "http://localhost:5000/" + gallery.path,
            'subject_gallery': gallery.subject_gallery,
            'images': getImagesForPerson(gallery.images),
            'avatar': gallery.images.first().path
        }
    return jsonify(empDict), 201

@app.route("/api/tasks/")
def task_overview():
    tasks = {}
    for key in redis.keys():
        if key == 'train_recognizer':
            content = train_recognizer.get_status()
            content['task_url'] = url_for('recognizer_training_status')
        else:
            content = json.loads(r.get(key))
        tasks[key] = content

    return jsonify(tasks), 201



@app.route("/models")
def models():
    return render_template('models.html', models=ClassifierStats.query.order_by(ClassifierStats.loaded.desc(),
                                                                                ClassifierStats.date.desc()).all())


@app.route("/api/resync")
def resync_db():
    sync_db_from_filesystem()
    return jsonify({'message': 'db resynced from filesystem'}), 201


@app.route('/images/<path:filename>')
def show_images(filename):
    return send_from_directory(config['IMAGE_BASE_PATH'], filename)


@app.route("/api/dlib_models/", methods=['GET', 'POST'])
def check_models():
    """
    Checks and downloads dlib_models. As of now the application expects the model files to exist on startup,
    so this method is useless for now...
    :return: 
    """
    # Check if Models exist
    if request.method == 'GET':
        if models_exist():
            return jsonify({'message': 'models found'})
        else:
            return jsonify({'message': 'one or more models missing'}), 409

    # Post Method, download models
    else:
        task = download_models.apply_async()
        task_url = url_for('model_download_status', task_id=task.id)
        return jsonify({'message': 'download started', 'location': task_url}), 202, {'Location': task_url}


@app.route("/api/dlib_models/status/<task_id>")
def model_download_status(task_id):
    task = download_models.AsyncResult(task_id)
    response = {
        'state': task.state,
        'current': task.info.get('current'),
        'total': task.info.get('total'),
        'status': task.info.get('status', ''),
        'model': task.info.get('model', '')
    }
    if 'result' in task.info:
        response['result'] = task.info['result']

    return jsonify(response)


@app.route("/teapot")
def teapot():
    recognition_manager.test()
    return jsonify({'message': 'I\'m a Teapot!'}), 418


@app.route("/status")
def status_recognition():
    status_arr = recognition_manager.get_status()
    # ["{}", "{}"]

    status_arr_dict = []
    for status in status_arr:
        status_arr_dict.append(json.loads(status))

    return jsonify(status_arr_dict), 200


@app.route("/api/recognizer/init")
def init_recognizer():
    if models_exist():
        recognizer.initialize(shape_predictor_path=config['DLIB_SHAPE_PREDICTOR_PATH'],
                              descriptor_model_path=config['DLIB_FACE_RECOGNITION_MODEL_PATH'])
        return jsonify({'message': 'Recognizer initalized'}), 201
    else:
        return jsonify(
            {'message': 'One ore more models are missing for the Recognizer to work.'}), 409

import threading

@app.route("/api/recognizer/train/")
def training_recognizer():
    taskid = train_recognizer.run()

    task_url = url_for('recognizer_training_status', task_id=taskid)
    return jsonify({'message': 'Training started', 'location': task_url}), 202, {'Location': task_url}
    #return jsonify({'message': 'Training started', }), 202


@app.route("/api/image/upload/", methods=['POST'])
def upload_image():
    parsed_args = image_upload_parser.parse_args()
    image = parsed_args['image']
    filename = str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')[:-3]) + '.jpg'
    id = new_image(image, filename)
    return jsonify({'message': 'Image uploaded'}), 200


@app.route("/api/live/", methods=['POST'])
def new_live_image():
    """
    Receives a image as base64 encoded string, saves it in the database and classifies it with the classify_db_image 
    method. After classification the result is emitted to clients via a socket io connection.
    :return: 
    """

    parsed_args = live_parser.parse_args()
    image = parsed_args['image']
    annotate = True
    if parsed_args['annotate'] is not None:
        if parsed_args['annotate'] == u'False':
            annotate = False
    filename = str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')[:-3]) + '.jpg'
    id = new_image(image, filename)

    latest_clf = ClassifierStats.query.order_by(ClassifierStats.date.desc()).first()
    if latest_clf:
        #result = classify_db_image(id)

        recognition_manager.add_image(id)

        #with heimdall.app_context():
        #    url = url_for('classify_db_image', image_id=id, _external=True)
        #r = requests.get(url)
        #result = redis.json()
        #if len(result['predictions']) > 0 and annotate:
        #    image = annotate_live_image(image, result)

        #print(result)

        #socketio.emit('new_image', json.dumps({'image': image,
        #                                    'image_id': id,
        #                                    'classification': result}))
        return jsonify({'message': 'Image processed'}), 200
    else:
        return jsonify({'message': 'No classifier present!'}), 500



@app.route("/api/recognizer/train/status")
def recognizer_training_status():
    return jsonify(train_recognizer.get_status()), 200


@app.route("/api/classifier/")
def get_classifier():
    return jsonify(ClassifierStats.query.order_by(ClassifierStats.loaded.desc(), ClassifierStats.date.desc()).all()), 200

@app.route("/api/classifier/load/", methods=['POST'])
def load_new_classifier():
    path = config['ML_MODEL_PATH'] + os.sep + '*.pkl'
    latest_model = max(glob.glob(path), key=os.path.getctime)
    app.clf = load_classifier(latest_model)
    db_model = ClassifierStats.query.order_by(ClassifierStats.date.desc()).first()
    app.labels = db_model.labels_as_dict()
    old_model = ClassifierStats.query.filter_by(loaded=True).first()
    if old_model:
        old_model.loaded = False
    db_model.loaded = True
    db.session.commit()

    return jsonify({'message': 'new model loaded into classifier'}), 201


@app.route("/api/classifier/load/<model_id>", methods=['POST'])
def load_db_classifier(model_id):
    model = ClassifierStats.query.filter_by(id=model_id).first()
    old_model = ClassifierStats.query.filter_by(loaded=True).first()
    old_model.loaded = False
    model.loaded = True
    app.clf = load_classifier(model.model_path)
    app.labels = model.labels_as_dict()
    db.session.commit()

    return jsonify({'message': 'new model loaded into classifier'}), 201


@app.route("/api/classifier/delete/<model_id>", methods=['DELETE'])
def delete_classifier_model(model_id):
    model = ClassifierStats.query.filter_by(id=model_id).first()
    if model is None:
        abort(409, descripction="Model not found.")
    if model.loaded:
        abort(409, descripction="Model is currently loaded, load other Model before deleting.")
    try:
        if os.path.isfile(model.model_path):
            os.remove(model.model_path)
        db.session.delete(model)

    except IOError:
        abort(409, description="Error occured while deleting model file from disk")
    db.session.commit()
    return jsonify({'message': 'model deleted'}), 200
