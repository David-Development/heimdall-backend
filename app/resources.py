from flask_restful import Resource, reqparse, marshal_with, fields, abort
from sqlalchemy.exc import IntegrityError
from flask import jsonify, send_from_directory, request, url_for
from models import Gallery, Image
from app import api, app, db
from tasks import sync_db_from_filesystem, delete_gallery, move_images, download_models
import os

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
def hello_world():
    return "hello_world"


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
    models_path = config['ML_MODEL_PATH']
    if request.method == 'GET':
        dlib_shape_predictor = False
        dlib_face_descriptor = False
        if os.path.exists(os.path.join(models_path, config['DLIB_SHAPE_PREDICTOR_MODEL'])):
            dlib_shape_predictor = True
        if os.path.exists(os.path.join(models_path, config['DLIB_FACE_RECOGNITION_MODEL'])):
            dlib_face_descriptor = True

        if dlib_shape_predictor and dlib_face_descriptor:
            return jsonify({'message': 'models found'})
        else:
            return jsonify({'message': 'one or more models missing'}), 409

    # Post Method, download models
    else:
        task = download_models.apply_async()
        task_url = url_for('taskstatus', task_id=task.id)
        return jsonify({'message': 'download started', 'location': task_url}), 202, {'Location': task_url}


@app.route("/status/<task_id>")
def taskstatus(task_id):
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
