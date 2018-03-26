import datetime
from heimdall.models.Gallery import Gallery
from heimdall.models.ClassificationResults import ClassificationResults
from heimdall.models.RecognitionResult import RecognitionResult
from heimdall.tasks import classify, extract_bounding_boxes
import numpy as np
import base64
import cv2


class Classification:

    @staticmethod
    def classify_db_image(classifier, classifier_stats, db_image_id, image, prob_threshold, labels, labels_gallery_dict):
        time_before_classification = datetime.datetime.now()

        results, bounding_boxes = classify(classifier, image)

        time_after_classification = datetime.datetime.now()
        diff = time_after_classification - time_before_classification

        classification_result = ClassificationResults(clf_id=classifier_stats.id,
                                                      image_id=db_image_id,
                                                      date=datetime.datetime.now())
        recognition_results = []

        # For each detected face
        for faces, bb in zip(results, bounding_boxes):
            # Save the highest probability (the result)
            highest = np.argmax(faces)
            prob = np.max(faces)

            # If the probability is less then the configures threshold, the person is unknown
            label = "unknown"
            if prob >= prob_threshold:
                label = labels[highest]

            gallery_id = labels_gallery_dict[label]

            print("Label:", label, " Gallery-ID:", gallery_id)
            recognition_results.append(RecognitionResult(classification=classification_result,
                                                         gallery_id=gallery_id,
                                                         probability=prob,
                                                         bounding_box=bb))

        with open("timings.txt", "a") as timing_file:
            text = str(datetime.datetime.now()) + " - Time needed for classification: " + str(diff) + \
                   " - Faces Count: " + str(len(bounding_boxes)) + "\n"
            timing_file.write(text)

        return classification_result, recognition_results


    @staticmethod
    def detect_faces(image):
        bounding_boxes = extract_bounding_boxes(image)
        print(bounding_boxes)
        return len(bounding_boxes) > 0
