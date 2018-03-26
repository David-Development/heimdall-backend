import os
import time
import sys
import concurrent.futures
import multiprocessing
import threading

import traceback
import json

from heimdall.app import mqtt
from heimdall.camera.camera import Camera
from heimdall.exceptions.ClassifierNotTrainedError import ClassifierNotTrainedError
from heimdall.recognition.Classification import Classification

from heimdall.profiler.line_profiler import line_profiler
from heimdall.recognition import utils
from heimdall.models.Image import Image
from heimdall.models.Gallery import Gallery
from heimdall.models.ClassifierStats import ClassifierStats
import base64
import cv2

from heimdall.recognition.debounce import debounce


app = None
db = None
last_recognized_annotated_image = None





queue = multiprocessing.Queue()
queue_results = multiprocessing.Queue()

class RecognitionManager:
    global app
    global db
    global queue
    global queue_results

    def __init__(self):
        print("init RecognitionManager")
        #with multiprocessing.Pool(processes=4) as pool:
        #    res = pool.apply_async(os.getpid, ())  # runs in *only* one process
        #    print(res.get(timeout=1))  # prints the PID of that process

        #self.queue = multiprocessing.Queue()
        self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=2)
        #self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        # with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        #   for num in fred:
        #       executor.submit(f, num)
        # the "with" keyword performs an implicit shutdown here

        self.handler_thread = threading.Thread(target=RecognitionManager.wait_for_results)
        self.handler_thread.daemon = True
        self.handler_thread.start()

    @debounce(120)  # time in seconds (120 = 2 minutes)
    def clear_mqtt_channels(self):
        print("Clear MQTT Channel!!!")
        mqtt.publish("recognitions/person", payload=None, qos=0, retain=True)
        mqtt.publish("recognitions/image", payload=None, qos=0, retain=True)

    @staticmethod
    #@line_profiler
    def wait_for_results():
        # counter = 0
        while True:
            (classification_result, recognition_results, image, db_image) = queue_results.get()

            queue.put(time.time())
            
            bbs = []
            predictions = []
            if recognition_results: # Empty sequences are false
                print("Publishing results")
                db.session.add(classification_result)

                names = []

                for rr in recognition_results:
                    db.session.add(rr)

                    highest_name = Gallery.query.filter(Gallery.id == rr.gallery_id).first().name
                    image = annotate_live_image(image, rr, highest_name)

                    names.append(highest_name)

                    bbs.append(rr.bounding_box)
                    prediction_result_dict = {'highest': highest_name,
                                              'bounding_box': rr.bounding_box,
                                              'probability': round(rr.probability, 4)}
                    predictions.append(prediction_result_dict)

                result = {'message': 'classification complete',
                          'predictions': predictions,
                          'bounding_boxes': bbs,
                          'img_path': db_image.path}

                result = json.dumps(result)

                mqtt.publish("recognitions/person", payload=result, qos=0, retain=True)
                recognition_manager.clear_mqtt_channels()

                '''
                names_string = ",".join(names)
                print("Names:", names_string)

                
                # Update user-variable
                domoticz = {'idx': 1,
                            'command': 'setuservariable',
                            'value': names_string }
                mqtt.publish("domoticz/in", payload=json.dumps(domoticz), qos=0, retain=True)

                # update virtual sensor
                domoticz = {'idx': 1,
                            'nvalue': 0,
                            'svalue': names_string }
                mqtt.publish("domoticz/in", payload=json.dumps(domoticz), qos=0, retain=True)
                
                mqtt.publish("recognitions/personname", payload=names_string, qos=0, retain=True)
                '''
            if not recognition_results and not classification_result:
                print("Skipping delete.. is the classifier trained yet?")
            elif not recognition_results:
                print("No face detected.. Deleting image")
                db.session.delete(db_image)

                storage_image_path = os.path.join(app.config['BASEDIR'], db_image.path)
                utils.delete_image(storage_image_path)

            db.session.commit()

            
            if recognition_results or classification_result is None: # Send if face was detected or if an error occured
                # Send image (as base64) via MQTT
                mqtt.publish("recognitions/image", payload=image_to_base64(image), qos=0, retain=True)
                
                # Send image (as raw jpeg) via MQTT
                #_, jpeg = cv2.imencode('.jpg', image)
                #mqtt.publish("recognitions/rawimage", payload=jpeg.tobytes(), qos=0, retain=True)
                Camera.load_image(image)
            elif classification_result is None:
                print("Not sending image - No classification result")
            else:
                print("Not sending image - Number of persons detected:", classification_result.num_persons)

            # Live View
            if(app.config['ENABLE_LIVEVIEW']):
                print("Note: Sending image to liveview - disable live view in production!")
                mqtt.publish("liveview", payload=image_to_base64(image), qos=0, retain=True)


    # The method below will be called on the (Thread/Process)-PoolExecutor
    @staticmethod
    def process_image(data):
        db_image, base_dir, prob_threshold, labels, labels_gallery_dict, classifier, classifier_stats = data[0]
        print("Job started! Image-ID:", db_image.id)
        storage_image_path = os.path.join(base_dir, db_image.path)
        image = utils.load_image(storage_image_path)

        classification_result = None
        recognition_results = []

        try:
            if image is None:
                raise AssertionError('Image was None / empty')

            if classifier_stats is None:
                raise ClassifierNotTrainedError('No model trained yet!')

            classification_result, recognition_results = Classification.classify_db_image(
                classifier=classifier,
                classifier_stats=classifier_stats,
                db_image_id=db_image.id,
                image=image,
                prob_threshold=prob_threshold,
                labels=labels,
                labels_gallery_dict=labels_gallery_dict
            )

        except ClassifierNotTrainedError:
            print("Classifier is not trained yet!")
        except Exception as e:
            print("Exception: ", e)
            print(traceback.format_exc())
        finally:
            if image is not None:
                queue_results.put((classification_result, recognition_results, image, db_image))
            else:
                print("Skipping image")


    def add_image(self, db_image):
        print("Scheduling image with ID:", db_image.id)

        #RecognitionManager.process_image([(image_id)])

        classifier_stats = ClassifierStats.query.order_by(ClassifierStats.date.desc()).first()

        labels_gallery_dict = {}
        for gallery in Gallery.query.all():
            labels_gallery_dict[gallery.name] = gallery.id

        self.executor.submit(RecognitionManager.process_image, [(db_image,
                                                                 app.config['BASEDIR'],
                                                                 app.config['PROBABILITY_THRESHOLD'],
                                                                 app.labels,
                                                                 labels_gallery_dict,
                                                                 app.clf,
                                                                 classifier_stats)])


    def get_times(self):
        times = []
        while not queue.empty():
            ts = queue.get()
            if ts:
                times.append(ts)
            else:
                break
        return times


recognition_manager = RecognitionManager()

def init(app1, db1):
    global app
    global db

    app = app1
    db = db1


def annotate_live_image(image, recognition_result, name):
    """
    annotates an image with bounding boxes, names and probabilities from the classification result
    :param image: the image to annotate encoded as base64.
    :param classification_result: a classification result with one or multiple faces.
    :return: a base64 encoded image with annotations
    """

    bb = recognition_result.bounding_box
    prob = recognition_result.probability
    if name == 'unknown':
        text = name
        color = (0, 0, 255)
    else:
        text = name + ': ' + str(int(round(prob*100))) + '%'
        color = (0, 255, 0)
    image = cv2.rectangle(image, pt1=(bb[0], bb[1]), pt2=(bb[0] + bb[2], bb[1] + bb[3]), color=color, thickness=1)
    image = cv2.putText(image, text, (bb[0], bb[1] + bb[3] + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness=2)

    return image


def image_to_base64(image):
    b64bytes = base64.b64encode(cv2.imencode('.jpg', image)[1])
    return b64bytes.decode("utf-8")
