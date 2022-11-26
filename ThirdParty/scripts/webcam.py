"""Video demo utils for showing live object detection from a camera

python video_demo.py --restore_weeights=weights/<weights.h5>

"""

import datetime
import cv2
import config
import json
import os
import tensorflow as tf
import numpy as np

import label_utils
from ssd_final import SSD
from ssd_parser import ssd_parser
from model import get_preprocess_fn, get_optimizer
from layer_utils import DecodeSSDPredictions
from label_utils import LabelUtil
from loss import Loss
from l2_normalization import L2Normalization
import encoder
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

class Video():
    def __init__(self,
    detector,
    camera=0,
    record="webcam"):
        self.tracking = True
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", 0.2, None)
        self.tracker = Tracker(metric, 0.6, 20, 6)
        
        self.camera = camera
        self.detector = detector
        self.encoder = encoder.create_box_encoder("resources/networks/mars-small128.pb", batch_size=1)

        self.record = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), record))
        self.preprocess_fn = get_preprocess_fn(self.detector.config["model"]["name"])

        self.capture = cv2.VideoCapture(self.camera) if record == "webcam" else cv2.VideoCapture(self.record)

        if not self.capture.isOpened():
            msg = "camera not found" if self.record == "webcam" else "file not found"
            raise Exception("Error opening the camera {}".format(msg))

        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def runtracking(self, image, y_pred):
        if self.tracking:
            y_pred_track = y_pred.copy()
            y_pred_track = np.array(y_pred_track)
            detections = []
            # print(y_pred_track[..., -2:])
            # print(y_pred_track[..., -4:-2])

            if(y_pred_track.shape[1]):
                y_pred_track[..., -2:] = y_pred_track[..., -2:] - y_pred_track[..., -4:-2]
                features = self.encoder(image, y_pred_track[..., 2:6])
                detections += [Detection(prediction[...,2:6], prediction[..., 1], feature) for prediction, feature in
                zip(y_pred_track[0], features)]

            self.tracker.predict()
            self.tracker.update(detections)

    def loop(self):
        font = cv2.FONT_HERSHEY_DUPLEX
        pos = (10, 30)
        font_scale = 0.9
        font_color = (0, 0, 0)
        line_type = 1
        confidence_threshold = 0.6

        input_height = self.detector.config["model"]["input_height"]
        input_width = self.detector.config["model"]["input_width"]

        while True:
            check, image = self.capture.read()

            if check is False:
                break
            
            image_height, image_width, _ = image.shape
            height_scale, width_scale = input_height/ image_height, input_width/image_width
            
            # image = cv2.resize(image, (input_height, input_width))
            image = cv2.resize(image, (300, 300))
            original_image = image.copy()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            start_time = datetime.datetime.now()

            y_pred = self.detector.predict(image)
            elapsed_time = datetime.datetime.now() - start_time
            hz = 1.0 / elapsed_time.total_seconds()
            # print("time: {}".format(1.0 / elapsed_time.total_seconds()))
            hz = "%0.2fHz" % hz

            # label_util = LabelUtil(self.detector.config, width_scale, height_scale)

            # if self.tracking:
            #     for track in self.tracker.tracks:
            #         label_util.show_tracks(original_image, track)
            # else:
            #     for prediction in y_pred[0]:
            #         label_util.show_labels(original_image, prediction, confidence_threshold)  

            y_pred = np.array(y_pred, dtype="int32")
            for prediction in y_pred[0]:
                cv2.rectangle(original_image, (prediction[1], prediction[2]), (prediction[3], prediction[4]), (0, 0, 255), 2)
                cv2.putText(original_image, "TRACK ID: " + str(prediction[0]), (prediction[1], prediction[2] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)

            cv2.putText(original_image,
                       hz,
                       pos,
                       font,
                       font_scale,
                       font_color,
                       line_type)
            
            cv2.imshow('image', original_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            

# if __name__ == '__main__':
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)

# tf.config.optimizer.set_jit(True)

parser = ssd_parser()

# help_ = "Camera index"
# parser.add_argument("--camera",
#                     default=0,
#                     type = int,
#                     help = help_)
# help_ = "Record video form webcam (record=\"webcam\") or enter filename"
# parser.add_argument("--record",
#                     default="webcam",
#                     help=help_)

# args = parser.parse_args()


ssd_model = SSD()
videodemo = Video(detector=ssd_model,
                    camera=0,
                    record="witcher3.avi")

videodemo.loop()
        # ssd_model.evaluate(image_file="000000180487.jpg")
