import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
import utils


class DefaultBoxes(Layer):
    """ A custom keras layer that generates default boxes for a given feature map.
    Args:
        - image_shape: The shape of the input image
        - scale: The current scale for the default box.
        - next_scale: The next scale for the default box.
        - aspect_ratios: The aspect ratios for the default boxes.
        - offset: The offset for the center of the default boxes. Defaults to center of each grid cell.
        - variances: The normalization values for each bounding boxes properties (cx, cy, width, height).
    Returns:
        - A tensor of shape (batch_size, feature_map_size, feature_map_size, num_default_boxes, 8)
    Raises:
        - feature map height does not equal to feature map width
        - image width does not equals to image height
    Code References:
        - https://github.com/pierluigiferrari/ssd_keras/blob/master/keras_layers/keras_layer_AnchorBoxes.py
    Paper References:
        - Liu, W., Anguelov, D., Erhan, D., Szegedy, C., Reed, S., Fu, C. Y., & Berg, A. C. (2016).
          SSD: Single Shot MultiBox Detector. https://arxiv.org/abs/1512.02325
    """

    def __init__(
        self,
        image_shape,
        scale,
        next_scale,
        aspect_ratios,
        variances,
        **kwargs
    ):
        self.image_shape = image_shape
        self.scale = scale
        self.next_scale = next_scale
        self.aspect_ratios = aspect_ratios
        self.variances = variances
        super(DefaultBoxes, self).__init__(**kwargs)

    def build(self, input_shape):
        # _, feature_map_height, feature_map_width, _ = input_shape
        # image_height, image_width, _ = self.image_shape

        # self.feature_map_size = min(feature_map_height,  feature_map_width)
        # self.image_size = min(image_height, image_width)
        self.feature_map_shape = input_shape
        super(DefaultBoxes, self).build(input_shape)

    def call(self, inputs):
        default_boxes = utils.generate_anchor_boxes_for_feature_map(
            feature_map_shape=inputs.shape,
            image_shape=self.image_shape,
            scale=self.scale,
            next_scale=self.next_scale,
            aspect_ratios=self.aspect_ratios,
            variances=self.variances)
        default_boxes = np.expand_dims(default_boxes, axis=0)
        default_boxes = tf.constant(default_boxes, dtype='float32')
        default_boxes = tf.tile(
            default_boxes, (tf.shape(inputs)[0], 1, 1, 1, 1))
        return default_boxes

    def get_config(self):
        config = {
            "image_shape": self.image_shape,
            "scale": self.scale,
            "next_scale": self.next_scale,
            "aspect_ratios": self.aspect_ratios,
            "variances": self.variances,
            # "feature_map_height": self.feature_map_shape[1],
            # "feature_map_width" : self.feature_map_shape[2],
            # "image_height": self.image_shape[1],
            # "image_width" : self.image_shape[2]
        }
        base_config = super(DefaultBoxes, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)

import time

def prediction_processing(
    y_pred,
    img_height,
    img_width,
    nms_max_output_size=100,
    confidence_threshold=0.5,
    iou_threshold=0.45,
    num_predictions=10):
    # class_id = tf.expand_dims(tf.compat.v1.to_float(tf.argmax(y_pred[...,:-12], axis=-1)), axis=-1)
    # scores = tf.reduce_max(y_pred[...,:-12], axis=-1, keepdims=True)

    cx = y_pred[..., -12] * y_pred[..., -4] * y_pred[..., -6] + y_pred[..., -8]
    cy = y_pred[...,-11] * y_pred[...,-3] * y_pred[...,-5] + y_pred[...,-7]
    w = tf.exp(y_pred[...,-10] * y_pred[...,-2]) * y_pred[...,-6]
    h = tf.exp(y_pred[...,-9] * y_pred[...,-1]) * y_pred[...,-5]

    xmin = (cx - 0.5 * w) * img_width
    ymin = (cy - 0.5 * h) * img_height
    xmax = (cx + 0.5 * w) * img_width
    ymax = (cy + 0.5 * h) * img_height
    
    batch_size = tf.shape(y_pred)[0]
    num_boxes = tf.shape(y_pred)[1]
    n_classes = tf.shape(y_pred)[2] - 4
    indices = tf.range(1, n_classes)

    y_pred_conv = np.copy(y_pred[..., -6:])
    y_pred_conv[..., 0] = np.argmax(y_pred[..., :-12], axis=-1)
    y_pred_conv[..., 1] = np.amax(y_pred[..., :-12], axis=-1)
    y_pred_conv[..., 2] = xmin
    y_pred_conv[..., 3] = ymin
    y_pred_conv[..., 4] = xmax  
    y_pred_conv[..., 5] = ymax

    def nmsfast(boxes_init, iou_threshold):
        boxes = np.copy(boxes_init)
        nms_boxes = []
        while boxes.shape[0]>0:
            nms_index = np.argmax(boxes[:, 1])
            nms_box = np.copy(boxes[nms_index])
            nms_boxes.append(nms_box)
            boxes = np.delete(boxes, nms_index, axis=0)
            if boxes.shape[0] == 0:
                break
            ious = utils.iou(boxes[:, 2:], nms_box[2:])
            boxes = boxes[ious<=iou_threshold]
        return np.array(nms_boxes)

        
    output = []
    for batch_item in y_pred_conv:
        boxes = batch_item[np.nonzero(batch_item[:,0])]
        boxes = boxes[boxes[..., 1] >= confidence_threshold]
        boxes = nmsfast(boxes, iou_threshold)
        boxes = np.array(boxes)
        if boxes.shape[0] > num_predictions:
            boxes = boxes[np.argpartition(boxes[:, 1], kth=boxes.shape[0]-num_predictions, axis=0)[boxes.shape[0]-num_predictions:]]
        output.append(boxes)

    return output

class DecodeSSDPredictions(Layer):
    def __init__(
        self,
        img_height,
        img_width,
        nms_max_output_size=100,
        confidence_threshold=0.5,
        iou_threshold=0.35,
        num_predictions=10,
        **kwargs
    ):
        self.img_height = img_height
        self.img_width = img_width
        self.nms_max_output_size = nms_max_output_size
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.num_predictions = num_predictions
        super(DecodeSSDPredictions, self).__init__(**kwargs)

    def build(self, input_shape):
        super(DecodeSSDPredictions, self).build(input_shape)

    def call(self, inputs):
        y_pred = prediction_processing(
            y_pred=inputs,
            img_height= self.img_height,
            img_width= self.img_width,
            nms_max_output_size=self.nms_max_output_size,
            confidence_threshold=self.confidence_threshold,
            iou_threshold=self.iou_threshold,
            num_predictions=self.num_predictions
        )
        return y_pred

    def get_config(self):
        config = {
            'img_height': self.img_height,
            'img_width' : self.img_width,
            'nms_max_output_size': self.nms_max_output_size,
            'confidence_threshold': self.confidence_threshold,
            'iou_threshold': self.iou_threshold,
            'num_predictions': self.num_predictions,
        }
        base_config = super(DecodeSSDPredictions, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)
