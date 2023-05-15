"""SSD network class
"""
import os
import numpy as np
import cv2
import json
import tensorflow as tf
from model import SSD_VGG16,  get_preprocess_fn, get_optimizer
from label_utils import get_label_dictionary, LabelUtil
from layer_utils import DecodeSSDPredictions, DefaultBoxes
from l2_normalization import L2Normalization
import datetime
# import cupy as cp
import tensorflow.experimental.dlpack as tfdlpack
from utils import iou
import encoder
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
class SSD():
    """SSD network class comprised of a backbone network and ssd head

    Arguments:
        -config: Pre-defined configurations for the SSD Model

    Returns:
        ssd(model) : An SSD network model   

    """

    def __init__(self):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.set_logical_device_configuration(gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=1024)])
        with open(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs\\configs.json"),), 'rb') as reader:
            config = json.load(reader)

        metric = nn_matching.NearestNeighborDistanceMetric("cosine", 0.2, None)
        self.tracker = Tracker(metric, 0.6, 8, 6)
        self.encoder = encoder.create_box_encoder("resources/networks/mars-small128.pb", batch_size=1)

        self.config = config
        self.name = config["model"]["name"]
        self.classnames = config["classes"]
        self.input_shape = (config["model"]["input_height"], config["model"]["input_width"], 3)
        self.confidence_threshold = 0.65
        # self.ssd = SSD_VGG16(config, self.classnames)
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
        filename = os.path.join(save_dir, "ssd_vgg16_model_fullfocal.h5")
        self.ssd = tf.keras.models.load_model(filename, custom_objects={'L2Normalization' : L2Normalization, 'DefaultBoxes' : DefaultBoxes}, compile=False)
        self.preprocess_fn = get_preprocess_fn(self.name)
        
        # self._restore_weights()

    def _restore_weights(self):
        """Load previously trained model weights"""
        save_dir = os.path.join(os.pathd.dirname(os.path.abspath(__file__)), "weights")
        filename = os.path.join(save_dir, "ssd_vgg16_fullfocal_weighted.h5")
        log = "Loading weights: %s" % filename
        print(log)
        self.ssd.load_weights(filename)
        # self.ssd.save(os.path.join(save_dir, "ssd_vgg16_model_fullfocal.h5"))

    def _runtracking(self, image, y_pred):
        y_pred_track = y_pred.copy()
        y_pred_track = np.array(y_pred_track)
        detections = []

        if (y_pred_track.shape[1]):
            y_pred_track[..., -2:] = y_pred_track[..., -2:] - y_pred_track[..., -4:-2]
            features = self.encoder(image, y_pred_track[..., 2:6])
            detections += [Detection(prediction[..., 2:6], prediction[..., 1], feature) for prediction, feature in
            zip(y_pred_track[0], features)]

        self.tracker.predict()
        self.tracker.update(detections)

    def _nms(self, detections):
        iou_threshold_tracking = 0.6
        boxes = np.copy(detections)
        nms_boxes = []
        while boxes.shape[0]>0:
            nms_index = np.argmax(boxes[:, -1])
            nms_box = np.copy(boxes[nms_index])
            nms_boxes.append(nms_box[:-1])
            boxes = np.delete(boxes, nms_index, axis=0)
            if boxes.shape[0] == 0:
                break
            ious = iou(boxes[:, 1:-1], nms_box[1:-1])
            boxes = boxes[ious<=iou_threshold_tracking]
        return np.array(nms_boxes)

    def predict(self, image):
        """Handles the prediction of the ssd network based on given input image"""
        input_height, input_width, _ = self.input_shape
        image_height, image_width, _ = image.shape
        height_scale, width_scale = input_height/ image_height, input_width/image_width
        
        original_image = image.copy()
        # image = cv2.resize(image, (input_width, input_height))
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if (self.preprocess_fn is not None):
            image = self.preprocess_fn(image)
        image = np.expand_dims(image, axis=0)
        # print(image)
        
        y_pred = self.ssd(image, training=False)
        y_pred = DecodeSSDPredictions(img_height=input_height,
                                    img_width=input_width,
                                    name="Decode Predictions",
                                    num_predictions=5,
                                    confidence_threshold=self.confidence_threshold,
                                    iou_threshold=0.4,
                                    dtype='float32')(y_pred)

        self._runtracking(original_image, y_pred)
        # y_pred = np.array(y_pred)
        # for prediction in y_pred[0]:
        #     prediction[2] = max(int(prediction[2] / width_scale), 1)
        #     prediction[3] = max(int(prediction[3] / height_scale), 1)
        #     prediction[4] = min(int(prediction[4] / width_scale), image_width - 1)
        #     prediction[5] = min(int(prediction[5] / height_scale), image_height - 1)
        # print(y_pred.dtype)
        # return np.array(y_pred)

        result = []
        for track in self.tracker.tracks:
            if (not track.is_confirmed()):
                continue
            xmin, ymin, xmax, ymax = track.to_tlbr().astype(np.int)
            xmin = max(int(xmin / width_scale), 1)
            ymin = max(int(ymin / height_scale), 1)
            xmax = min(int(xmax / width_scale), image_width - 1)
            ymax = min(int(ymax / height_scale), image_height - 1)
            result.append([track.track_id, int(xmin), ymin, xmax, ymax, track.age])
        result = np.array(result)
        result = self._nms(result)
        result = np.expand_dims(result, axis=0)
        ####
        # result = np.array([[1, 200, 500, 300, 700]])
        # result = np.expand_dims(result, axis=0)
        # print(result)
        # print(result.dtype)
        return np.array(result, dtype='float32')

    def predict_gpu(self, ptr, shape, size, strides):
        k = 1
        input_height, input_width, _ = self.input_shape
        image_height, image_width, _ = (300, 300, 3)
        height_scale, width_scale = input_height/ image_height, input_width/image_width        
        mem = cp.cuda.memory.MemoryPointer(cp.cuda.memory.UnownedMemory(ptr, size, k), 0)
        image = cp.ndarray(shape, cp.uint8, mem, strides)
        image = cp.asarray(cp.ascontiguousarray(image)).toDlpack()
        image = tfdlpack.from_dlpack(image)
        image = tf.convert_to_tensor(image)
        image = tf.cast(image, tf.float32)

        if (self.preprocess_fn is not None):
            image = self.preprocess_fn(image)
        image = tf.expand_dims(image, axis=0)
        
        y_pred = self.ssd(image, training=False)
        y_pred = DecodeSSDPredictions(img_height=input_height,
                                    img_width=input_width,
                                    name="Decode Predictions",
                                    num_predictions=5,
                                    confidence_threshold=self.confidence_threshold,
                                    iou_threshold=0.4,
                                    dtype='float32')(y_pred)

        self._runtracking(image, y_pred)

        # for prediction in y_pred[0]:
        #     prediction[2] = max(int(prediction[2] / width_scale), 1)
        #     prediction[3] = max(int(prediction[3] / height_scale), 1)
        #     prediction[4] = min(int(prediction[4] / width_scale), image_width - 1)
        #     prediction[5] = min(int(prediction[5] / height_scale), image_height - 1)
        # return np.array(y_pred)

        result = []
        for track in self.tracker.tracks:
            xmin, ymin, xmax, ymax = track.to_tlbr().astype(np.int)
            xmin = max(int(xmin / width_scale), 1)
            ymin = max(int(ymin / height_scale), 1)
            xmax = min(int(xmax / width_scale), image_width - 1)
            ymax = min(int(ymax / height_scale), image_height - 1)
            result.append([track.track_id, xmin, ymin, xmax, ymax])

        return np.array(result, dtype='float32')