import os
import shutil
from urllib.request import urlopen
import bz2
import time
from datetime import datetime, timedelta
import multiprocessing
import base64
import gc

import requests
from flask import url_for, json
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.externals import joblib
import numpy as np
import cv2

from heimdall.app import db, recognizer, clf, redis, app
from heimdall.models.Gallery import Gallery
from heimdall.models.Image import Image
from heimdall.models.ClassifierStats import ClassifierStats
from heimdall.models.Labels import Labels
from heimdall.models.Event import Event
from heimdall.models.ClassificationResults import ClassificationResults

from heimdall.recognition import utils, augmenter

import scikitplot
import matplotlib.pyplot as plt


config = app.config

def sync_db_from_filesystem():
    """
    Resynchronizes the filesystem with the database. CAUTION: All galleries and images in the database are deleted!
    :return: 
    """
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
    """
    create a gallery and the images in the gallery folder in the database
    :param base_path: the filesystem path for the gallery
    :param gallery_name: the name for the gallery
    :param subject_gallery: indicates the gallery contains images from a single person (True) or not (False)
    :return: 
    """

    event = Event.query.first()
    if event is None:
        event = Event()
        db.session.add(event)
        db.session.commit()

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
            db.session.add(Image(name=file_path, event_id=event.id, gallery_id=gallery.id, path=os.path.join(gallery_folder, file_path)))
    db.session.commit()


def new_image(image, filename):
    """
    create a new image on the filesystem and database
    :param image: the image as base64 encoded string
    :param filename: the filename
    :return: the image id 
    """
    path = os.path.join(config['NEW_IMAGES_PATH'], filename)
    new_gallery = Gallery.query.filter_by(name='new').first()

    event_time_frame = datetime.now() - timedelta(minutes=1)
    print("event_time_frame:", event_time_frame)
    event = Event.query.filter(Event.begindate>=event_time_frame).first()

    if event is None:
        print("No matching event found.. creating new one..")
        event = Event()
        db.session.add(event)
        db.session.commit()
    else:
        print("Found matching event!")

    print("Using event:", event.begindate)

    with open(path, 'wb') as f:
        f.write(base64.b64decode(image))
    image = Image(name=filename, event_id=event.id, gallery_id=new_gallery.id, path=os.path.join(config['NEW_IMAGES_FOLDER'], filename))
    db.session.add(image)
    db.session.commit()

    return image.id


def clear_files_from_db():
    """
    clear images and galleries from the database, only for a resync
    :return: 
    """
    clear_table(ClassificationResults)
    clear_table(Image)
    clear_table(Gallery)
    db.session.commit()


def clear_table(model):
    """
    clear the given model table
    :param model: the model to clear
    :return: 
    """
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
    """
    move images from one gallery to another
    :param src_gallery: source gallery
    :param dest_gallery: destination gallery
    :return: 
    """
    basedir = config['BASEDIR']
    src_path = os.path.join(basedir, src_gallery.path)
    dest_path = os.path.join(basedir, dest_gallery.path)

    for file_path in os.listdir(src_path):
        shutil.move(os.path.join(src_path, file_path), os.path.join(dest_path, file_path))


def clear_gallery(gallery):
    """
    Removes all images from the gallery
    :param gallery: the gallery to be cleared
    :return: 
    """
    basedir = config['BASEDIR']
    for image in gallery.images:
        ClassificationResults.query.filter_by(image=image).delete()
        if os.path.isfile(os.path.join(basedir, image.path)):
            os.remove(os.path.join(basedir, image.path))

    gallery.images = []

    db.session.commit()


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


def download_models(self):
    """
    Downloads multiple pretrained models for dlib
    :return: 
    """

    path_dlib_shape_predictor_model  = os.path.join(config['ML_MODEL_PATH'], config['DLIB_SHAPE_PREDICTOR_MODEL'])
    path_dlib_face_recognition_model = os.path.join(config['ML_MODEL_PATH'], config['DLIB_FACE_RECOGNITION_MODEL'])
    print("Download: " + config['DLIB_SHAPE_PREDICTOR_MODEL_URL'] + " to " + path_dlib_shape_predictor_model)
    print("Download: " + config['DLIB_FACE_RECOGNITION_MODEL']    + " to " + path_dlib_face_recognition_model)

    if not os.path.isfile(path_dlib_shape_predictor_model):
        self.update_state(state='STARTED', meta={'current': 1, 'total': 2, 'status': 'Downloading Dlib Shape Predictor'})

        response = urllib2.urlopen(config['DLIB_SHAPE_PREDICTOR_MODEL_URL'])
        model = chunk_read(response, self, name="Dlib Shape Predictor", report_hook=chunk_report)

        with open(path_dlib_shape_predictor_model, 'w') as f:
            self.update_state(state='STARTED',
                              meta={'current': 1, 'total': 2, 'status': 'Unpacking Dlib Shape Predictor'})
            model = bz2.decompress(model)
            f.write(model)

    if not os.path.isfile(path_dlib_face_recognition_model):
        self.update_state(state='STARTED',
                          meta={'current': 2, 'total': 2, 'status': 'Downloading Dlib Face Descriptor Model'})
        response = urllib2.urlopen(config['DLIB_FACE_RECOGNITION_MODEL_URL'])
        model = chunk_read(response, self, name="Dlib Face Descriptor Model", report_hook=chunk_report)

        with open(path_dlib_face_recognition_model, 'w') as f:
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
    """
    Checks if the necessary models for Dlib exist.
    :return: 
    """
    models_path = config['ML_MODEL_PATH']
    dlib_shape_predictor = False
    dlib_face_descriptor = False
    if os.path.exists(os.path.join(models_path, config['DLIB_SHAPE_PREDICTOR_MODEL'])):
        dlib_shape_predictor = True
    if os.path.exists(os.path.join(models_path, config['DLIB_FACE_RECOGNITION_MODEL'])):
        dlib_face_descriptor = True

    return dlib_shape_predictor and dlib_face_descriptor


from multiprocessing import Process
import random

class TrainRecognizer:

    def __init__(self):
        self.task_id = random.randint(0, 1000000)
        redis.delete("train_recognizer")

    def get_status(self):
        status =redis.get("train_recognizer")
        if status:
            return json.loads(status)
        return {}

    def run(self):
        self.update_status('IDLE', {})

        p = Process(target=self.train_recognizer)
        # p = Process(target=self.train_recognizer, args=(taskid,))
        p.daemon = True
        p.start()

        return self.task_id

    def update_status(self, state, meta):
        content = json.dumps({'task_id': self.task_id, 'status': {'state': state, 'meta': meta}})
        print("Content:", content)
        # redis.hset("train_recognizer", self.task_id, content)
        # redis.hmset("train_recognizer", content)
        redis.set("train_recognizer", content)

    def train_recognizer(self, clf_type="SVM", n_jobs=-1, k=5, cross_val=True):
        """
        Trains a new model
        :param self: the celery instance that runs this task
        :param taskid: taskid
        :param clf_type: type of the classifier, "SVM" or "kNN" (untested for the latter)
        :param n_jobs: number of processes to use for training of the classifier. -1 to use number of cpu cores
        :param k: only for "kNN", number of nearest neighbors
        :param cross_val: if cross validation should be performed for the classifier
        :return:
        """

        classifier = create_classifier(clf_type, n_jobs, k)
        X, y, folder_names = utils.load_dataset(config['SUBJECTS_BASE_PATH'], grayscale=False)
        avg_images = np.mean(np.unique(y, return_counts=True)[1])

        start = time.time()
        #X, y = self.augment_images(X, y, target=config['NUM_TARGET_IMAGES'])
        X, y = self.augment_images(X, y, folder_names, target=config['NUM_TARGET_IMAGES'])

        label_dict = utils.create_label_dict(y, folder_names)
        total_images = len(y)
        i = 0
        no_face = 0
        cv_score = None
        transformed = []
        labels = []

        print("_______________________________")
        print("Num of Pictures:", len(X))
        print("Num of Labels:", len(y))
        print("Unique Labels:", np.unique(y))
        print("Label Dictionary:", label_dict)

        for data in zip(X, y):
            image = data[0]
            label = data[1]
            i += 1
            self.update_status('STARTED', {'current': i, 'total': total_images, 'step': 'Transforming'})
            descriptors, _ = recognizer.extract_descriptors(image)
            if len(descriptors) != 0:
                for descriptor in descriptors:
                    transformed.append(descriptor)
                    labels.append(label)

                #imgOutput = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # convert colors back
                #cv2.imwrite("./images_for_debugging/face_" + str(i) + ".jpg", imgOutput)
            else:
                print("No face at position:", i)
                print("Label:", label)
                no_face += 1

                imgOutput = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # convert colors back
                cv2.imwrite("./images_for_debugging/no_face_" + str(i) + ".jpg", imgOutput)

        if cross_val:
            self.update_status('STARTED', {'current': total_images, 'total': total_images,'step': 'Scoring'})
            cv_score = np.mean(cross_val_score(classifier, transformed, labels, cv=5, n_jobs=n_jobs))

            self.update_status('STARTED',{'current': total_images, 'total': total_images,'step': 'Training'})
        classifier.fit(transformed, labels)

        print("Labels after zip:", np.unique(labels))
        print("Transformed after (len):", len(transformed))
        print("Classifier:", classifier)
        print("No face count:", no_face)

        training_time = time.time() - start
        timestamp = time.strftime('%Y%m%d%H%M%S')
        filename = clf_type + timestamp
        full_filename = filename + '.pkl'
        path = os.path.join(config['ML_MODEL_PATH'], full_filename)

        predictions = classifier.predict(transformed)
        scikitplot.metrics.plot_confusion_matrix(y_true=labels, y_pred=predictions)
        confusion_path = os.path.join(config['PLOTS_BASE_PATH'], filename + '_confusion.png')
        plt.savefig(confusion_path, bbox_inches='tight')

        # scikitplot.estimators.plot_learning_curve(classifier, transformed, labels) # TODO this line is not working!
        learning_curve_path = os.path.join(config['PLOTS_BASE_PATH'], filename + '_learning.png')
        plt.savefig(learning_curve_path, bbox_inches='tight')

        stats = ClassifierStats(name=clf_type + timestamp, classifier_type=clf_type, model_path=path,
                                date=datetime.now(), cv_score=cv_score, total_images=total_images,
                                total_no_face=no_face, training_time=training_time, avg_base_img=avg_images,
                                num_classes=len(folder_names), confusion_matrix=confusion_path,
                                learning_curve=learning_curve_path)
        db.session.add(stats)
        # flush to generate stats id
        db.session.flush()

        for key, value in label_dict.items():
            label_entry = Labels(clf_id=stats.id, num=key, label=value)
            db.session.add(label_entry)

        db.session.commit()
        save_classifier(classifier, path)

        with app.app_context():
            if config['DOCKER']:
                url = "http://heimdall:5000/api/classifier/load/"
            else:
                url = url_for('load_new_classifier', _external=True)
            requests.post(url)

        del X, y, transformed
        gc.collect()

        content = {'task_id': self.task_id, 'status': 'FINISHED'}
        redis.hmset("train_recognizer", content)

        #r.delete("train_recognizer")

        return {'current': total_images, 'total': total_images, 'step': 'Training',
                'result': 'Training finished'}

    def augment_images(self, images, labels, folder_names, target):
        """
        Augments the images up to a certain number
        :param images: Images to augment
        :param labels: Corresponding labels
        :param target: The target number of images
        :param celery_binding: the celery binding object, for updating the status
        :return: The augmented images and corresponding labels
        """
        unq, unq_inv, unq_cnt = np.unique(labels, return_inverse=True, return_counts=True)

        print("___________________________________")
        print("augment_images...")
        print("Labels:", labels)
        print("Labels Unique:", unq)
        print("Labels Unique Invers:", unq_inv)
        print("Labels Unique Count:", unq_cnt)

        # count the number of images that already exist for each user
        unique_class_indices = np.split(np.argsort(unq_inv), np.cumsum(unq_cnt[:-1]))

        target_x = images
        target_y = labels
        i = 0
        for unique_class in unq:
            print("Unique Class:", unique_class)
            indices = unique_class_indices[unique_class]
            #print("Unique Class Indeces:", indices)

            folder_name = folder_names[unique_class]
            print("Gallery:", folder_name)

            number_of_missing_images = target - len(indices)
            self.update_status('STARTED', {'current': int(unique_class + 1), 'total': len(unique_class_indices),
                                  'step': 'Augmenting'})

            print("Number of missing Images: ", number_of_missing_images)
            batch_x = augmenter.augment_array_target(folder_name, number_of_missing_images)
            batch_y = np.full(number_of_missing_images, unique_class)
            print(" ")
            print("batch_y:", batch_y)

            target_x.extend(batch_x)
            target_y.extend(batch_y)
            i += 1
        return target_x, target_y


def create_classifier(clf_type="SVM", n_jobs=-1, k=5):
    """
    Creates a new classifier from Scikit-Learn.
    :param clf_type: The classifier type. Either "SVM" or "kNN"
    :param n_jobs: Number of processes (only used in "kNN"). -1 for number of cpu cores
    :param k: Number of neighbors for "kNN"
    :return: The untrained classifier object
    """
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()

    if clf_type is "kNN":
        classifier = KNeighborsClassifier(k=k, n_jobs=n_jobs)
    else:
        classifier = SVC(C=10, gamma=1, kernel='rbf', probability=True)

    return classifier


def load_classifier(path):
    """
    Load and deserialize a trained classifier from the filesystem
    :param path: The filepath
    :return: The deserialized classifier
    """
    return joblib.load(path)


def save_classifier(classifier, path):
    """
    serialize and save a trained classifier to the filesystem
    :param classifier: the classifier to save
    :param path: the filepath
    :return: 
    """
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
    """
    classify a image on the given classifier
    :param classifier: a Scikit-Learn classifier with the predict_proba method available.
    :param image: The image to be classified
    :param dists: For "kNN" as classifier, if a distance is given, return absolute distances to number of neighbors
    :param neighbors: For "kNN" as classifier, the number of neighbors for which a distance should be returned
    :return: Returns the probabilities or distances for each persons for each face found in the image
    """
    descriptors, bbs = recognizer.extract_descriptors(image)

    results = []
    for descriptor in descriptors:

        if isinstance(classifier, KNeighborsClassifier) and dists:
            results.append(classifier.kneighbors(neighbors))
        else:
            descriptor = np.asarray(descriptor)
            descriptor = descriptor.reshape(1, -1)
            results.append(classifier.predict_proba(descriptor)[0])

    return results, bbs




# TODO remove method below!
#@app.context_processor
def annotate_processor():
    """
    create context processors for use in jinja2 templates
    :return: 
    """

    def annotate_db_image(classification_result):
        image = cv2.imread(os.path.join(config['BASEDIR'], classification_result.image.path))

        for result in classification_result.results:
            bb = result.bounding_box
            name = result.gallery.name
            prob = result.probability

            if name == 'unknown':
                text = name
                color = (0, 0, 255)
            else:
                text = name + ': ' + str(round(prob, 2))
                color = (0, 255, 0)
            image = cv2.rectangle(image, pt1=(bb[0], bb[1]), pt2=(bb[0] + bb[2], bb[1] + bb[3]), color=color,
                                  thickness=1)
            image = cv2.putText(image, text, (bb[0], bb[1] + bb[3] + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness=2)

        return image_to_base64(image)

    return dict(annotate_db_image=annotate_db_image)


def image_to_base64(image):
    b64bytes = base64.b64encode(cv2.imencode('.jpg', image)[1])
    return b64bytes.decode("utf-8")