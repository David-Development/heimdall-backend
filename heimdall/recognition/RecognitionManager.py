import os
import time
import sys
import redis
import concurrent.futures
import multiprocessing

import traceback
import json

from heimdall.app import mqtt

app = None
db = None


'''
print("RecognitionManager __init__ called!")
print("###################################")
for line in traceback.format_stack():
    line = line.strip()
    if str(line).startswith("File \"<frozen"):
        pass
    else:
        print("Hierarchy: " + line)
print("###################################")
'''


'''
class RecognitionManager:
    def __init__(self):
        print("init RecognitionManager")
        self.queue = multiprocessing.Queue()
        self.pool = multiprocessing.Pool(3, self.worker, (self.queue,))

    def add_image(self, image):
        self.queue.put(image)

    def worker(self, queue):
        print(os.getpid(), "working")
        while True:
            print(os.getpid(), " - Waiting!")
            item = self.queue.get(True)
            print(os.getpid(), " - Waiting!")
            print(os.getpid(), "got", item)
'''


class RecognitionManager:
    global app
    global db

    def __init__(self):
        print("init RecognitionManager")
        #with multiprocessing.Pool(processes=4) as pool:
        #    res = pool.apply_async(os.getpid, ())  # runs in *only* one process
        #    print(res.get(timeout=1))  # prints the PID of that process

        self.redis = redis.StrictRedis(host="redis", port="6379", charset="utf-8", decode_responses=True)
        #self.queue = multiprocessing.Queue()
        #self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=3)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)

        # fred = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        print("Check server..")
        self.check_server()
        print("Clear server..")
        self.redis.delete("test")
        # print("Submit jobs..")
        # for num in fred:
        #    self.executor.submit(process_image, num)
        # print("Shutdown..")
        # self.executor.shutdown()
        # print("Print results..")
        # print(self.redis.lrange("test", 0, -1))


        # with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        #   for num in fred:
        #       executor.submit(f, num)
        # the "with" keyword performs an implicit shutdown here

    def check_server(self):
        try:
            self.redis.info()
        except redis.exceptions.ConnectionError:
            print("Error: cannot connect to redis server. Is the server running?")
            sys.exit(1)

    @staticmethod
    def process_image(data):
        try:
            print("Job started!")
            redis, image_id = data[0]
            # print("------------")
            # print("App: ", heimdall)
            # print("DB: ", db)
            # print("Config: ", config)
            # print("Image id:", image_id)
            # print("Image: ", image[:10])
            # print("------------")

            db_image = Image.query.filter_by(id=image_id).first()
            image_path = os.path.join(app.config['BASEDIR'], db_image.path)
            image = utils.load_image(image_path)

            result = classify_db_image(db_image.id, image)

            annotate = True
            if len(result['predictions']) > 0 and annotate:
                image = annotate_live_image(image, result)



            # print("Image Path: ", image_path)
            cv2.imwrite('/live_view.jpg', image)
            Camera.currentImage = Camera.load_image('/live_view.jpg')

            # TODO mqtt new image
            #socketio.emit('new_image', json.dumps({'image': image_to_base64(image),
            #                                       'image_id': image_id,
            #                                       'classification': result}))

            #print("Result: ", result)
            result["img_path"] = "http://localhost:5000/" + db_image.path
            result = json.dumps(result)

            mqtt.publish("recognitions/person", payload=result, qos=0, retain=True)
            #mqtt.publish("recognitions/image", payload=image_to_base64(image), qos=0, retain=True)

            redis.rpush("test", result)
        except Exception as e:
            print("Exception: ", e)

    def add_image(self, image_id):
        print("Scheduling!")

        #print(heimdall)
        #print(socketio)
        #print(db)
        print(image_id)

        self.executor.submit(RecognitionManager.process_image, [(self.redis, image_id)])
        # print(redis_threading.lrange("test", 0, -1))

    def test(self):
        print(self.get_status())

    def get_status(self):
        return self.redis.lrange("test", 0, -1)

    '''
    def add_image(self, image):
        self.queue.put(image)

    def worker(self, queue):
        print(os.getpid(), "working")
        while True:
            print(os.getpid(), " - Waiting!")
            item = self.queue.get(True)
            print(os.getpid(), " - Waiting!")
            print(os.getpid(), "got", item)
    '''


recognition_manager = RecognitionManager()


def init(app1, db1):
    global app
    global db

    app = app1
    db = db1
    pass





import datetime
from heimdall.recognition import utils
from heimdall.models.Gallery import Gallery
from heimdall.models.Image import Image
from heimdall.models.ClassifierStats import ClassifierStats
from heimdall.models.ClassificationResults import ClassificationResults
from heimdall.models.RecognitionResult import RecognitionResult
from heimdall.tasks import classify
import numpy as np
import base64
import cv2


def classify_db_image(db_image_id, image):
    time_before_classification = datetime.datetime.now()

    results, bbs = classify(app.clf, image)

    time_after_classification = datetime.datetime.now()
    diff = time_after_classification - time_before_classification

    latest_clf = ClassifierStats.query.order_by(ClassifierStats.date.desc()).first()
    classification_result = ClassificationResults(clf_id=latest_clf.id, image_id=db_image_id,
                                                  date=datetime.datetime.now())
    db.session.add(classification_result)
    db.session.flush()

    predictions = []
    # For each detected face
    for faces, bb in zip(results, bbs):
        # Save the highest probability (the result)
        highest = np.argmax(faces)
        prob = np.max(faces)

        # If the probability is less then the configures threshold, the person is unknown
        if prob < app.config['PROBABILITY_THRESHOLD']:
            label = 'unknown'
        else:
            label = app.labels[highest]
        gallery = Gallery.query.filter_by(name=label).first()
        db.session.add(
            RecognitionResult(classification=classification_result.id, gallery_id=gallery.id, probability=prob, bounding_box=bb))
        prediction_dict = {}
        prediction_result_dict = {'highest': label,
                                  'bounding_box': bb,
                                  'probability': round(prob, 4)}
        # All probabilities
        for idx, prediction in enumerate(faces):
            prediction_dict[app.labels[idx]] = round(prediction, 4)
        prediction_result_dict['probabilities'] = prediction_dict

        predictions.append(prediction_result_dict)

        db.session.commit()

    with open("timings.txt", "a") as timing_file:
        text = str(datetime.datetime.now()) + " - Time needed for classification: " + str(diff) + \
               " - Faces Count: " + str(len(bbs)) + "\n"
        timing_file.write(text)

    return {'message': 'classification complete',
                    'predictions': predictions,
                    'bounding_boxes': bbs}


def annotate_live_image(image, classification_result):
    """
    annotates an image with bounding boxes, names and probabilities from the classification result
    :param image: the image to annotate encoded as base64.
    :param classification_result: a classification result with one or multiple faces.
    :return: a base64 encoded image with annotations
    """
    #image = base64.b64decode(image)
    #image = np.fromstring(image, dtype=np.uint8)
    #image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    for prediction in classification_result['predictions']:
        bb = prediction['bounding_box']
        name = prediction['highest']
        prob = prediction['probability']
        if name == 'unknown':
            text = name
            color = (0, 0, 255)
        else:
            text = name + ': ' + str(round(prob, 2))
            color = (0, 255, 0)
        image = cv2.rectangle(image, pt1=(bb[0], bb[1]), pt2=(bb[0] + bb[2], bb[1] + bb[3]), color=color, thickness=1)
        image = cv2.putText(image, text, (bb[0], bb[1] + bb[3] + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness=2)

    return image


def image_to_base64(image):
    b64bytes = base64.b64encode(cv2.imencode('.jpg', image)[1])
    return b64bytes.decode("utf-8")
