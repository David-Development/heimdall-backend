import os
import glob

from flask_restful import Resource, reqparse, marshal_with, fields, abort
from sqlalchemy.exc import IntegrityError
from flask import jsonify, send_from_directory, request, url_for, render_template
import numpy as np
from flask_socketio import send, emit

from models import Gallery, Image, ClassifierStats
from app import api, app, db, recognizer, clf, labels, socketio
from recognition import utils
from tasks import (sync_db_from_filesystem, delete_gallery, move_images, download_models, models_exist,
                   train_recognizer, load_classifier, classify, run_camera_socket)

config = app.config

gallery_parser = reqparse.RequestParser()
gallery_parser.add_argument('name')

image_parser = reqparse.RequestParser()
image_parser.add_argument('image_ids', action="append")
image_parser.add_argument('gallery_id')

gallery_fields = {
    'id': fields.Integer,
    'name': fields.String,
    'path': fields.String,
    'images': fields.Integer(attribute='images_count'),
    'subject_gallery': fields.Boolean,
}

image_fields = {
    'id': fields.Integer,
    'name': fields.String,
    'path': fields.String,
    'url': fields.String,
    'gallery_id': fields.Integer(attribute='gallery.id'),
    'gallery_name': fields.String(attribute='gallery.name'),
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
        db.session.commit()

        return {'message': 'images moved'}, 200


class GalleryImagesListRes(Resource):
    @marshal_with(image_fields)
    def get(self, gallery_id):
        images = Image.query.filter_by(gallery_id=gallery_id).all()
        return images


api.add_resource(GalleryRes, '/api/gallery/', '/api/gallery/<gallery_id>/')
api.add_resource(GalleryImagesListRes, '/api/gallery/<gallery_id>/images/')
api.add_resource(GalleryListRes, '/api/galleries/')
api.add_resource(ImageListRes, '/api/images/')


@app.route("/")
def index():
    return render_template('index.html', title="Heimdall Face Recognition")


@app.route("/liveview")
def liveview():
    return render_template('liveview.html', title="Liveview")


@app.route("/api/resync")
def resync_db():
    sync_db_from_filesystem()
    return jsonify({'message': 'db resynced from filesystem'}), 201


@app.route('/images/<path:filename>')
def show_images(filename):
    return send_from_directory(config['IMAGE_BASE_PATH'], filename)


@app.route("/api/models/", methods=['GET', 'POST'])
def check_models():
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


@app.route("/api/models/status/<task_id>")
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
    return jsonify({'message': 'I\'m a Teapot!'}), 418


@app.route("/api/recognizer/init")
def init_recognizer():
    if models_exist():
        recognizer.initialize(shape_predictor_path=config['DLIB_SHAPE_PREDICTOR_PATH'],
                              descriptor_model_path=config['DLIB_FACE_RECOGNITION_MODEL_PATH'])
        return jsonify({'message': 'Recognizer initalized'}), 201
    else:
        return jsonify(
            {'message': 'One ore more models are missing for the Recognizer to work.'}), 409


@app.route("/api/recognizer/train/")
def training_recognizer():
    task = train_recognizer.apply_async()
    task_url = url_for('recognizer_training_status', task_id=task.id)
    return jsonify({'message': 'Training started', 'location': task_url}), 202, {'Location': task_url}


@app.route("/api/recognizer/classify/<image_id>")
def classify_db_image(image_id):
    image = Image.query.filter_by(id=image_id).first()
    image_path = os.path.join(config['BASEDIR'], image.path)
    image = utils.load_image(image_path)

    results, bbs = classify(app.clf, image)

    predictions = []
    # Jedes erkannte Gesicht
    for faces, bb in zip(results, bbs):
        # Das wahrscheinlichste Ergebnis
        highest = np.argmax(faces)
        prob = np.max(faces)
        prediction_dict = {}
        prediction_result_dict = {'highest': app.labels[highest],
                                  'bounding_box': bb,
                                  'probability': round(prob, 4)}
        # Alle Wahrscheinlichkeiten
        for idx, prediction in enumerate(faces):
            prediction_dict[app.labels[idx]] = round(prediction, 4)
        prediction_result_dict['probabilities'] = prediction_dict

        predictions.append(prediction_result_dict)

    return jsonify({'message': 'classification complete',
                    'predictions': predictions,
                    'bounding_boxes': bbs})


@app.route("/api/recognizer/train/status/<task_id>")
def recognizer_training_status(task_id):
    task = train_recognizer.AsyncResult(task_id)
    response = {
        'state': task.state,
        'current': task.info.get('current'),
        'total': task.info.get('total'),
        'step': task.info.get('step', ''),
        'model': task.info.get('model', '')
    }
    if 'result' in task.info:
        response['result'] = task.info['result']

    return jsonify(response)


@app.route("/api/classifier/load/", methods=['POST'])
def load_new_classifier():
    path = config['ML_MODEL_PATH'] + os.sep + '*.pkl'
    latest_model = max(glob.glob(path), key=os.path.getctime)
    app.clf = load_classifier(latest_model)
    db_model = ClassifierStats.query.order_by(ClassifierStats.date.desc()).first()
    app.labels = db_model.labels_as_dict()

    return jsonify({'message': 'new model loaded into classifier'}), 201


@app.route("/api/cameralistener/", methods=['POST'])
def start_camera_listener():
    run_camera_socket.apply_async()
    return jsonify({'message': 'socket started'}), 202


@socketio.on('connect')
def connect_live_view():
    send('connected')


@socketio.on('disconnect')
def disconnect_live_view():
    send('disconnected')
