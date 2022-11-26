import cv2
import os
import numpy as np
import tensorflow as tf
import skimage
import utils
import generator_utils
from tensorflow.keras.utils import Sequence
import data_augmentation
import copy


class DataGenerator(Sequence):
    """Data Generator for training SSD networks using with VOC labeled format.

    Args:
        - samples : A list of string representing a data sample (img path + label file path)
        - config : python dictionary as read from the config json file
        - label_map : A list of classes in the dataset
        - shuffle : Boolean that denotes whether or not to shuffle the batch
        - batch_size : The size of each batch
        - augment : Boolean that denotes whether or not to perform data augmentation on the training samples
        - preproccess_input_fn : A function to preprocess the input image(s) before feeding them to SSD network

    """

    def __init__(
            self,
            args,
            dictionary,
            config,
            label_map,
            shuffle,
            batch_size,
            augment,
            process_input_function):
        training_config = config["training"]
        model_config = config["model"]
        self.args = args
        self.model_name = model_config["name"]
        self.match_threshold = training_config["match_threshold"]
        self.neutral_threshold = training_config["neutral_threshold"]
        self.default_boxes_config = model_config["default_boxes"]
        self.dictionary = dictionary
        self.keys = np.array(list(dictionary.keys()))
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.label_map = label_map
        self.n_classes = len(self.label_map)
        self.indices = range(0, len(self.dictionary))

        assert self.batch_size <= len(
            self.indices), "batch size must be smaller than the number of samples"
        self.input_height = model_config["input_height"] 
        self.input_width = model_config["input_width"]
        self.input_template = self.__get_input_template()
        self.augment = augment
        self.process_input_fn = process_input_function
        self.on_epoch_end()

    def __len__(self):
        return len(self.indices) // self.batch_size

    def __on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.keys)

    def __getitem__(self, index):
        batch = self.keys[index *
                          self.batch_size: (index + 1) * self.batch_size]
        x, y = self.__get_data(batch)
        return x, y

    def __get_input_template(self):
        if "min_scale" in self.default_boxes_config:
            scales = np.linspace(
                self.default_boxes_config["min_scale"],
                self.default_boxes_config["max_scale"],
                len(self.default_boxes_config["layers"])
            ) 
        else:
            scales = self.default_boxes_config["scales"]

        if "aspect_ratios" in self.default_boxes_config:
            aspect_ratios = self.default_boxes_config["aspect_ratios"]
        else:
            aspect_ratios = [layer["aspect_ratios"] for layer in self.default_boxes_config["layers"]] 

        mbox_conf_layers = []
        mbox_loc_layers = []
        mbox_anchors_layers = []
        for i, layer in enumerate(self.default_boxes_config["layers"]):
            layer_anchors = utils.generate_anchor_boxes_for_feature_map(
                feature_map_shape=(None, layer["size"], layer["size"], 3),
                image_shape=(self.input_height, self.input_width, 3),
                scale=scales[i],
                next_scale=scales[i+1] if i +
                1 <= len(self.default_boxes_config["layers"]) - 1 else 1,
                aspect_ratios=aspect_ratios[i],
                variances=self.default_boxes_config["variances"]
            )
            layer_anchors = np.reshape(layer_anchors, (-1, 8))
            layer_conf = np.zeros((layer_anchors.shape[0], self.n_classes))
            layer_conf[:, 0] = 1  # all classes are background by default
            mbox_conf_layers.append(layer_conf)
            mbox_loc_layers.append(np.zeros((layer_anchors.shape[0], 4)))
            mbox_anchors_layers.append(layer_anchors)
        mbox_conf = np.concatenate(mbox_conf_layers, axis=0)
        mbox_loc = np.concatenate(mbox_loc_layers, axis=0)
        mbox_anchors = np.concatenate(mbox_anchors_layers, axis=0)
        template = np.concatenate([mbox_conf, mbox_loc, mbox_anchors], axis=-1)
        template = np.expand_dims(template, axis=0)
        return np.tile(template, (self.batch_size, 1, 1))

    def __augment(self, image, boxes, classes):
        augment_simple = [
            data_augmentation.apply_random_brightness,
            data_augmentation.apply_random_contrast,
            data_augmentation.apply_random_hue,
            #data_augmentation.apply_random_noise,
            data_augmentation.apply_random_saturation
        ]

        new_img, new_boxes, new_classes = image, boxes, classes
        for aug in augment_simple:
            new_img = aug(image=new_img)

        # for aug in augment_img_box:
        #     new_img, new_boxes = aug(new_img, new_boxes)

        return new_img, new_boxes, new_classes

    def __get_data(self, batch):
        x = []
        y = self.input_template.copy()

        for i, key in enumerate(batch):
            img_path = os.path.join(
                self.args.data_path, "images\\train2017", key)
            img_path = os.path.normpath(img_path)

            image = cv2.imread(img_path)
            image = np.array(image, dtype=np.float)
            labels = np.array(self.dictionary[key])
            bboxes = np.array(labels[:, :4], dtype=np.float)
            classes = labels[:, -1]

            if self.augment:
                image, bboxes, classes = self.__augment(
                    image=image,
                    boxes=bboxes,
                    classes=classes)

            image_height, image_width, _ = image.shape
            height_scale, width_scale = self.input_height/image_height, self.input_width/image_width
            input_img = cv2.resize(
                np.uint8(image), (self.input_width, self.input_height))
            input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
            if self.process_input_fn is not None:
                input_img = self.process_input_fn(input_img)

            gt_classes = np.zeros((bboxes.shape[0], self.n_classes))
            gt_boxes = np.zeros((bboxes.shape[0], 4))
            anchors = y[i, :, -8:]

            for bbox_i in range(bboxes.shape[0]):
                bbox = bboxes[bbox_i]
                cx = ((bbox[0] + 0.5 * bbox[2]) * width_scale) / self.input_width
                cy = ((bbox[1] + 0.5 * bbox[3]) * height_scale) / self.input_height
                width = (bbox[2] * width_scale) / self.input_width
                height = (bbox[3] * height_scale) / self.input_height
                gt_boxes[bbox_i] = [cx, cy, width, height]
                gt_classes[bbox_i] = generator_utils.one_hot_class_label(
                    classes[bbox_i], self.label_map)

            matches, neutral_boxes = generator_utils.match_boxes(
                gt_boxes=gt_boxes,
                anchors=anchors[:, :4],
                match_threshold=self.match_threshold,
                neutral_threshold=self.neutral_threshold
            )
            y[i, matches[:, 1], self.n_classes: self.n_classes + 4] = gt_boxes[matches[:, 0]]
            y[i, matches[:, 1], 0: self.n_classes] = gt_classes[matches[:, 0]]
            # set neutral ground truth boxes to default boxes with appropriate class
            y[i, neutral_boxes[:, 1], self.n_classes: self.n_classes +4] = gt_boxes[neutral_boxes[:, 0]]
            y[i, neutral_boxes[:, 1], 0: self.n_classes] = np.zeros((self.n_classes))  # neutral boxes have a class vector of all zeros
            
            # encode the bounding boxes
            y[i] = generator_utils.encode_boxes(y[i])
            x.append(input_img)

        x = np.array(x, dtype=np.float)
        
        return x, y

class DataGeneratorSSD7(Sequence):
    """Data Generator for training SSD networks using with VOC labeled format.

    Args:
        - samples : A list of string representing a data sample (img path + label file path)
        - config : python dictionary as read from the config json file
        - label_map : A list of classes in the dataset
        - shuffle : Boolean that denotes whether or not to shuffle the batch
        - batch_size : The size of each batch
        - augment : Boolean that denotes whether or not to perform data augmentation on the training samples
        - preproccess_input_fn : A function to preprocess the input image(s) before feeding them to SSD network

    """

    def __init__(
            self,
            args,
            dictionary,
            config,
            label_map,
            feature_shapes,
            shuffle,
            batch_size,
            augment):
        training_config = config["training"]
        model_config = config["model"]
        self.args = args
        self.model_name = model_config["name"]
        self.match_threshold = training_config["match_threshold"]
        self.neutral_threshold = training_config["neutral_threshold"]
        self.default_boxes_config = model_config["default_boxes"]
        self.dictionary = dictionary
        self.keys = np.array(list(dictionary.keys()))
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.feature_shapes = feature_shapes
        self.label_map = label_map
        self.n_classes = len(self.label_map)
        self.indices = range(0, len(self.dictionary))

        assert self.batch_size <= len(
            self.indices), "batch size must be smaller than the number of samples"
        self.input_height = model_config["input_height"]
        self.input_width = model_config["input_width"]
        self.input_template = self.__get_input_template()
        self.augment = augment
        self.on_epoch_end()

    def __len__(self):
        return len(self.indices) // self.batch_size

    def __on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.keys)

    def __getitem__(self, index):
        batch = self.keys[index *
                          self.batch_size: (index + 1) * self.batch_size]
        x, y = self.__get_data(batch)
        return x, y


    def __get_input_template(self):
        scales = self.default_boxes_config["scales"]
        aspect_ratios = self.default_boxes_config["aspect_ratios"]
        mbox_conf_layers = []
        mbox_loc_layers = []
        mbox_anchors_layers = []
        for i, feat_shape in enumerate(self.feature_shapes):
            layer_anchors = utils.generate_anchor_boxes_for_feature_map(
                feature_map_shape=feat_shape,
                image_shape=(self.input_height, self.input_width, 3),
                scale=scales[i],
                next_scale=scales[i+1],
                aspect_ratios=aspect_ratios[i],
                variances=self.default_boxes_config["variances"]
            )
            layer_anchors = np.reshape(layer_anchors, (-1, 8))
            layer_conf = np.zeros((layer_anchors.shape[0], self.n_classes))
            layer_conf[:, 0] = 1  # all classes are background by default
            mbox_conf_layers.append(layer_conf)
            mbox_loc_layers.append(np.zeros((layer_anchors.shape[0], 4)))
            mbox_anchors_layers.append(layer_anchors)
        mbox_conf = np.concatenate(mbox_conf_layers, axis=0)
        mbox_loc = np.concatenate(mbox_loc_layers, axis=0)
        mbox_anchors = np.concatenate(mbox_anchors_layers, axis=0)
        template = np.concatenate([mbox_conf, mbox_loc, mbox_anchors], axis=-1)
        template = np.expand_dims(template, axis=0)
        return np.tile(template, (self.batch_size, 1, 1))

    def __augment(self, image, boxes, classes):
        augment_simple = [
            data_augmentation.apply_random_brightness,
            data_augmentation.apply_random_contrast,
            data_augmentation.apply_random_hue,
            #data_augmentation.apply_random_noise,
            data_augmentation.apply_random_saturation
        ]
        new_img, new_boxes, new_classes = image, boxes, classes
        for aug in augment_simple:
            new_img = aug(image=new_img)

        # for aug in augment_img_box:
        #     new_img, new_boxes = aug(new_img, new_boxes)

        return new_img, new_boxes, new_classes

    def __get_data(self, batch):
        x = []
        y = self.input_template.copy()

        for i, key in enumerate(batch):
            img_path = os.path.join(self.args.data_path, "images\\train2017", key)
            img_path = os.path.normpath(img_path)

            image = cv2.imread(img_path)
            image = np.array(image, dtype=np.float)


            labels = np.array(self.dictionary[key])
            bboxes = np.array(labels[:, :4], dtype=np.float)

            classes = labels[:, -1]

            if self.augment:
                image, bboxes, classes = self.__augment(
                    image=image,
                    boxes=bboxes,
                    classes=classes)

            image_height, image_width, _ = image.shape
            height_scale, width_scale = self.input_height/image_height, self.input_width /image_width
            input_img = cv2.resize(np.uint8(image), (self.input_width, self.input_height))
            input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
            # input_img = self.process_input_fn(input_img)

            gt_classes = np.zeros((bboxes.shape[0], self.n_classes))
            gt_boxes = np.zeros((bboxes.shape[0], 4))
            anchors = y[i, :, -8:]

            for bbox_i in range(bboxes.shape[0]):
                bbox = bboxes[bbox_i]
                cx = ((bbox[0] + 0.5 * bbox[2]) * width_scale) / self.input_width
                cy = ((bbox[1] + 0.5 * bbox[3]) * height_scale) / self.input_height
                width = (bbox[2] * width_scale) / self.input_width
                height = (bbox[3] * height_scale) / self.input_height
                gt_boxes[bbox_i] = [cx, cy, width, height]
                gt_classes[bbox_i] = generator_utils.one_hot_class_label(
                    classes[bbox_i], self.label_map)

            matches, neutral_boxes = generator_utils.match_boxes(
                gt_boxes=gt_boxes,
                anchors=anchors[:, :4],
                match_threshold=self.match_threshold,
                neutral_threshold=self.neutral_threshold
            )
            y[i, matches[:, 1], self.n_classes: self.n_classes + 4] = gt_boxes[matches[:, 0]]
            # set class scores label
            y[i, matches[:, 1], 0: self.n_classes] = gt_classes[matches[:, 0]]
            # set neutral ground truth boxes to default boxes with appropriate class
            y[i, neutral_boxes[:, 1], self.n_classes: self.n_classes + 4] = gt_boxes[neutral_boxes[:, 0]]
            y[i, neutral_boxes[:, 1], 0: self.n_classes] = np.zeros((self.n_classes))  # neutral boxes have a class vector of all zeros
            # encode the bounding boxes
            y[i] = generator_utils.encode_boxes(y[i])
            x.append(input_img)

        x = np.array(x, dtype=np.float)
        
        return x, y
