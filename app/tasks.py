import os
import shutil
import urllib2
import bz2
import time

import multiprocessing

import requests
from flask import url_for
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
import numpy as np

from app import db, app, celery, recognizer
from models import Gallery, Image

from recognition import utils

config = app.config


def sync_db_from_filesystem():
    clear_files_from_db()
    # start with new and unknown images
    create_and_populate_gallery(config['IMAGE_BASE_PATH'], 'new', False)
    # create_and_populate_gallery(config['SUBJECTS_BASE_PATH'], 'unknown', False)

    # and the subjects
    for gallery_folder in os.listdir(config['SUBJECTS_BASE_PATH']):
        if os.path.isdir(os.path.join(config['SUBJECTS_BASE_PATH'], gallery_folder)):
            if gallery_folder == 'unknown':
                create_and_populate_gallery(config['SUBJECTS_BASE_PATH'], gallery_folder, False)
            else:
                create_and_populate_gallery(config['SUBJECTS_BASE_PATH'], gallery_folder)


def create_and_populate_gallery(base_path, gallery_name, subject_gallery=True):
    gallery_path = os.path.join(base_path, gallery_name)

    if subject_gallery:
        gallery_folder = os.path.join(config['SUBJECT_IMAGES_FOLDER'], gallery_name)
    else:
        gallery_folder = os.path.join(config['IMAGE_FOLDER'], gallery_name)

    gallery = Gallery(name=gallery_name, path=gallery_folder, subject_gallery=subject_gallery)
    db.session.add(gallery)
    db.session.commit()
    for file_path in os.listdir(gallery_path):
        _, ext = os.path.splitext(file_path)
        if str(ext).lower() in ['.png', '.jpg', '.jpeg']:
            db.session.add(Image(name=file_path, gallery_id=gallery.id, path=os.path.join(gallery_folder, file_path)))
    db.session.commit()


def clear_files_from_db():
    clear_table(Image)
    clear_table(Gallery)
    db.session.commit()


def clear_table(model):
    db.session.query(model).delete()


def delete_gallery(src_gallery, dest_gallery):
    """
    Move Images from source to destination gallery and removes src gallery
    :param src_gallery: the source gallery :type Gallery
    :param dest_gallery: the destination gallery :type Gallery
    :return: 
    """

    # move images
    move_gallery_content(src_gallery=src_gallery, dest_gallery=dest_gallery)

    # delete src gallery
    os.rmdir(os.path.join(config['BASEDIR'], src_gallery.path))


def move_gallery_content(src_gallery, dest_gallery):
    basedir = config['BASEDIR']
    src_path = os.path.join(basedir, src_gallery.path)
    dest_path = os.path.join(basedir, dest_gallery.path)

    for file_path in os.listdir(src_path):
        shutil.move(os.path.join(src_path, file_path), os.path.join(dest_path, file_path))


def move_images(gallery, images):
    """
    Move on or more images into a new gallery
    :param gallery: the new gallery for the images
    :param images: the images to be moved
    :return: 
    """

    basedir = config['BASEDIR']
    dest_gallery_path = os.path.join(basedir, gallery.path)

    for image in images:
        image_path = os.path.join(basedir, image.path)
        shutil.move(image_path, os.path.join(dest_gallery_path, image.name))


@celery.task(bind=True)
def download_models(self):
    """
    Downloads multiple pretrained models for dlib
    :return: 
    """

    self.update_state(state='STARTED', meta={'current': 1, 'total': 2, 'status': 'Downloading Dlib Shape Predictor'})
    print config['DLIB_SHAPE_PREDICTOR_MODEL_URL']
    response = urllib2.urlopen(config['DLIB_SHAPE_PREDICTOR_MODEL_URL'])
    model = chunk_read(response, self, name="Dlib Shape Predictor", report_hook=chunk_report)

    with open(os.path.join(config['ML_MODEL_PATH'], config['DLIB_SHAPE_PREDICTOR_MODEL']), 'w') as f:
        self.update_state(state='STARTED',
                          meta={'current': 1, 'total': 2, 'status': 'Unpacking Dlib Shape Predictor'})
        model = bz2.decompress(model)
        f.write(model)

    self.update_state(state='STARTED',
                      meta={'current': 2, 'total': 2, 'status': 'Downloading Dlib Face Descriptor Model'})
    response = urllib2.urlopen(config['DLIB_FACE_RECOGNITION_MODEL_URL'])
    model = chunk_read(response, self, name="Dlib Face Descriptor Model", report_hook=chunk_report)

    with open(os.path.join(config['ML_MODEL_PATH'], config['DLIB_FACE_RECOGNITION_MODEL']), 'w') as f:
        self.update_state(state='STARTED',
                          meta={'current': 2, 'total': 2, 'status': 'Unpacking Dlib Face Descriptor Model'})
        model = bz2.decompress(model)
        f.write(model)

    return {'current': 2, 'total': 2, 'status': 'Task completed!', 'result': 'Models downloaded'}


def chunk_report(bytes_so_far, chunk_size, total_size, self, name):
    percent = float(bytes_so_far) / total_size
    percent = round(percent * 100, 2)
    self.update_state(state='STARTED',
                      meta={'current': percent, 'total': total_size, 'model': name, 'status': 'Downloading...'})
    # sys.stdout.write("Downloaded %d of %d bytes (%0.2f%%)\r" %
    #                 (bytes_so_far, total_size, percent))

    # if bytes_so_far >= total_size:
    #    sys.stdout.write('\n')


def chunk_read(response, self, name, chunk_size=1024 * 256, report_hook=None):
    total_size = response.info().getheader('Content-Length').strip()
    total_size = int(total_size)
    bytes_so_far = 0
    data = ''

    while 1:
        chunk = response.read(chunk_size)
        bytes_so_far += len(chunk)
        data += chunk

        if not chunk:
            break

        if report_hook:
            report_hook(bytes_so_far, chunk_size, total_size, self, name)

    return data


def models_exist():
    models_path = config['ML_MODEL_PATH']
    dlib_shape_predictor = False
    dlib_face_descriptor = False
    if os.path.exists(os.path.join(models_path, config['DLIB_SHAPE_PREDICTOR_MODEL'])):
        dlib_shape_predictor = True
    if os.path.exists(os.path.join(models_path, config['DLIB_FACE_RECOGNITION_MODEL'])):
        dlib_face_descriptor = True

    return dlib_shape_predictor and dlib_face_descriptor


@celery.task(bind=True)
def train_recognizer(self, clf_type="SVM", n_jobs=-1, k=5):
    classifier = create_classifier(clf_type, n_jobs, k)
    X, y, folder_names = utils.load_dataset(config['SUBJECTS_BASE_PATH'], grayscale=False)
    i = 0
    no_face = 0
    transformed = []
    labels = []
    for data in zip(X, y):
        image = data[0]
        label = data[1]
        i += 1
        descriptors, _ = recognizer.extract_descriptors(image)
        if len(descriptors) != 0:
            for descriptor in descriptors:
                transformed.append(descriptor)
                labels.append(label)
        else:
            no_face += 1

    classifier.fit(transformed, labels)

    timestamp = time.strftime('%Y%m%d%H%M%S')
    filename = clf_type + timestamp + '.pkl'
    save_classifier(classifier, os.path.join(config['ML_MODEL_PATH'], filename))
    with app.app_context():
        url = url_for('load_new_classifier', _external=True)
    print url
    requests.post(url)

    return {'current_image': len(X), 'total_images': len(X), 'step': 'Training',
            'result': 'Training finished'}


def training_progress_hook(current_image, total_images, training, celery_binding):
    step = "Generating descriptors..."
    if training:
        step = "Training..."

    celery_binding.update_state(state='STARTED',
                                meta={'current_image': current_image, 'total_images': total_images,
                                      'step': step})


def create_classifier(clf_type="SVM", n_jobs=-1, k=5):
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()

    if clf_type is "kNN":
        classifier = KNeighborsClassifier(k=k, n_jobs=n_jobs)
    else:
        classifier = SVC(C=10, gamma=1, kernel='rbf', probability=True)

    return classifier


def load_classifier(path):
    return joblib.load(path)


def save_classifier(classifier, path):
    joblib.dump(classifier, path)


def grid_search(X, y, classifier, param_grid, n_jobs=-1):
    """
    Perform gridsearch for internal classifier and apply best parameters
    :param X: Array of feature vectors 
    :param y: Corresponding labels
    :param classifier: The classifier 
    :param n_jobs: Number of processes to use for the gridsearch
    :param param_grid: appropriate parameter grid for the internal classifier
    Example for SVM:
    kernels = ['linear', 'rbf']
    Cs = [0.001, 0.01, 0.1, 1, 10, 100]
    gammas = [0.001, 0.01, 0.1, 1, 10, 100]
    param_grid = {'C': Cs, 'gamma': gammas, 'kernel': kernels}
    :return: 
    """
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
    grid_cv = GridSearchCV(classifier, param_grid=param_grid, n_jobs=n_jobs)
    grid_cv.fit(X, y)
    return grid_cv.best_estimator_


def classify(classifier, image, dists=False, neighbors=None):
    descriptors, bbs = recognizer.extract_descriptors(image)

    results = []
    for descriptor in descriptors:
        if isinstance(classifier, KNeighborsClassifier) and dists:
            results.append(classifier.kneighbors(neighbors))
        else:
            descriptor = np.asarray(descriptor)
            descriptor.reshape(1, -1)
            results.append(classifier.predict_proba(descriptor))

    return results, bbs
