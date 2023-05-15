"""SSD network class
"""
import os
import numpy as np
import cv2
import json
import tensorflow as tf
from tensorflow.keras.applications import vgg16
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.engine.node import KerasHistory
from loss import Loss
from model import SSD_VGG16,  get_preprocess_fn, get_optimizer
from data_generator import DataGenerator
from label_utils import get_label_dictionary, LabelUtil
from layer_utils import DecodeSSDPredictions
from ssd_parser import ssd_parser
from matplotlib import pyplot as plt
from math import ceil
import utils
from tensorflow.keras.backend import clear_session
import datetime
from layer_utils import DefaultBoxes
from l2_normalization import L2Normalization


class SSD():
    """SSD network class comprised of a backbone network and ssd head

    Arguments:
        -args : User defined arguments

    Returns:
        ssd(model) : An SSD network model

    """

    def __init__(self, config, args):
        self.args = args
        self.config = config
        self.name = config["model"]["name"]
        self.classnames = config["classes"]
        self.ssd = SSD_VGG16(config, self.classnames)
        self.input_shape = (
            config["model"]["input_height"], config["model"]["input_width"], 3)

        self.traindims = {}
        self.valdims = {}
        with open('traindict.txt', 'rb') as reader:
             self.traindims = json.loads(reader.read())
        with open('valdict.txt', 'rb') as reader:
             self.valdims = json.loads(reader.read())

    def __build_dictionary(self, setname):
        filepath = os.path.normpath(
            "{}/annotations/instances_{}.json".format(self.args.data_path, setname))
        return get_label_dictionary(filepath=filepath, labels=self.classnames)

    def _conditional(self, dim, element):
        # img_path = os.path.join(
        #     self.args.data_path, "images\\train2017", image_file)
        # img_path = os.path.normpath(img_path)
        # image = cv2.imread(img_path)

        org_height, org_width = dim[0], dim[1]
        width_scale, height_scale = self.input_shape[1]/org_width, self.input_shape[0]/org_height

        new_width, new_height = int(element[2] * width_scale), int(element[3] * height_scale)

        return (new_width * new_height) > 2500

    def train(self):

        training_config = self.config["training"]
        if not os.path.isdir(self.args.save_dir):
            os.makedirs(self.args.save_dir)

        dictionary = self.__build_dictionary(setname="train2017")
        dictionary_val = self.__build_dictionary(setname="val2017")

        #keep more "difficult images"
        # for key in list(dictionary.keys()):
        #     if len(dictionary[key])<4:
        #         del dictionary[key]
        # for key in list(dictionary_val.keys()):
        #     if len(dictionary_val[key])<4:
        #         del dictionary_val[key]

        # for key in list(dictionary.keys()):
        #     dim = (self.traindims[key][1], self.traindims[key][0]) #file is in width height format

        #     dictionary[key] = [e for e in dictionary[key] if self._conditional(dim, e)]
        #     if len(dictionary[key]) == 0:
        #         del dictionary[key]

        # for key in list(dictionary_val.keys()):
        #     dim = (self.valdims[key][1], self.valdims[key][0]) #file is in width height format

        #     dictionary_val[key] = [e for e in dictionary_val[key] if self._conditional(dim, e)]
        #     if len(dictionary_val[key]) == 0:
        #         del dictionary_val[key]

        self.train_generator = DataGenerator(self.args,
                                             dictionary,
                                             self.config,
                                             label_map=self.classnames,
                                             shuffle=True,
                                             batch_size=self.args.batch_size,
                                             augment=True,
                                             process_input_function=get_preprocess_fn(self.name))
        self.val_generator = DataGenerator(self.args,
                                           dictionary_val,
                                           self.config,
                                           label_map=self.classnames,
                                           shuffle=True,
                                           batch_size=self.args.batch_size,
                                           augment=True,
                                           process_input_function=get_preprocess_fn(self.name))

        savedir = os.path.join(os.getcwd(), self.args.save_dir)
        modelname = self.name + "_fullfocal_1.h5"
        filepath = os.path.join(savedir, modelname)

        checkpoint = ModelCheckpoint(filepath=filepath,
                                     verbose=1,
                                     monitor="val_loss" if self.args.validate is True else "loss",
                                     mode='auto',
                                     save_best_only=True,
                                     save_weights_only=False)

        csv_logger = CSVLogger(filename='training_logs\\ssd_focal_training_log.csv',
                               separator=',',
                               append=True)

        early_stopping = EarlyStopping(monitor='val_loss',
                                       min_delta=0.0,
                                       patience=12,
                                       verbose=1)

        reduce_learning_rate = ReduceLROnPlateau(monitor='val_loss',
                                                 factor=0.2,
                                                 patience=8,
                                                 verbose=1,
                                                 min_lr=0.00001)

        ssd_loss = Loss(alpha=training_config["alpha"],
                        min_negative_boxes=training_config["min_negative_boxes"],
                        negative_boxes_ratio=training_config["negative_boxes_ratio"])
        
        optimizer = get_optimizer(config, args)

        self.ssd.compile(optimizer=optimizer, loss=ssd_loss.focal_loss)

        callbacks = [checkpoint,
                     reduce_learning_rate,
                     csv_logger,
                     early_stopping,
                     ]

        self.ssd.fit(
            x=self.train_generator,
            validation_data=self.val_generator,
            batch_size=self.args.batch_size,
            validation_batch_size=self.args.batch_size,
            epochs=self.args.epochs,
            initial_epoch=self.args.initial_epoch,
            steps_per_epoch=self.args.steps,
            validation_steps=ceil(0.2 * self.args.steps),
            callbacks=callbacks,
            use_multiprocessing=False,
            workers=self.args.workers)

        # self.ssd = tfmot.sparsity.keras.strip_pruning(self.ssd)
        # tf.keras.models.save_model(self.ssd, filepath = os.path.join(savedir, self.name + "_focal_pruning_strip.h5"))

    def restore_weights(self):
        """Load previously trained model weights"""
        # if self.args.restore_weights:
        #     save_dir = os.path.join(os.getcwd(), self.args.save_dir)
        #     filename = os.path.join(save_dir, self.args.restore_weights)
        #     log = "Loading weights: %s" % filename
        #     print(log, self.args.verbose)
        #     self.ssd.load_weights(filename, by_name=True)
        restore_weights = "ssd_vgg16_fullfocal_weighted.h5"
        save_dir = os.path.join(os.getcwd(), self.args.save_dir)
        filename = os.path.join(save_dir, restore_weights)
        log = "Loading weights: %s" % filename
        print(log, self.args.verbose)
        self.ssd.load_weights(filename, by_name=True)

    def evaluate(self, image=None, image_file=None):
        """Evaluation function for the pre-trained SSD model"""
        if image_file is None:
            image = cv2.imread(image)
        else:
            img_path = os.path.join(
                self.args.data_path, "images\\train2017", image_file)
            img_path = os.path.normpath(img_path)
            image = cv2.imread(img_path)

        validation_dictionary = self.__build_dictionary(setname="val2017")
        keys = np.array(list(validation_dictionary.keys()))

        model_config = self.config['model']
        input_height = model_config['input_height']
        input_width = model_config['input_width']

        image = np.array(image, np.float)
        image = np.uint8(image)
        original_img = image.copy()

        img_height, img_width, _ = image.shape
        h_scale, w_scale = input_height / img_height, input_width/img_width

        image = cv2.resize(image, (input_width, input_height))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        preprocess_fn = get_preprocess_fn(config["model"]["name"])
        image = preprocess_fn(image)
        image = np.expand_dims(image, axis=0)
        y_pred = self.ssd.predict(image)
        y_pred = DecodeSSDPredictions(
            img_height=input_height,
            img_width=input_width,
            name="Decoded Predictions",
            num_predictions=5,
            confidence_threshold=self.args.confidence_threshold,
            iou_threshold=self.args.threshold
        )(y_pred)

        label_util = LabelUtil(
            config=self.config, width_scale=w_scale, height_scale=h_scale)

        for prediction in y_pred[0]:
            label_util.show_labels(
                original_img, prediction, self.args.confidence_threshold, print_box=True)

        if (image_file is not None):
            labels = np.array(validation_dictionary[image_file])
            print(labels.shape)
            bboxes = np.ones((labels.shape[0], 6))
            bboxes[:, 2:6] = np.array(labels[:, :4], dtype=np.float)
            bboxes[:, 4] += bboxes[:, 2]
            bboxes[:, 5] += bboxes[:, 3]
            for bbox in bboxes:
                label_util.show_labels(
                    original_img, bbox, self.args.confidence_threshold, ground_truth=True, print_box=True)

        cv2.imshow('image', original_img)
        cv2.waitKey(0)

    def evaluate_test(self):
        "Evaluates model on specific evaluation dataset"

        validation_dictionary = self.__build_dictionary(setname="val2017")
        
        for key in list(validation_dictionary.keys()):
            dim = (self.valdims[key][1], self.valdims[key][0]) #file is in width height format

            validation_dictionary[key] = [e for e in validation_dictionary[key] if self._conditional(dim, e)]
            if len(validation_dictionary[key]) == 0:
                del validation_dictionary[key]
        keys = np.array(list(validation_dictionary.keys()))
        size = len(keys)

        input_size = self.config["model"]["input_height"]
        # width_scale, height_scale = input_size / image.shape[1], input_size / image.shape[0]
        preprocess_fn = get_preprocess_fn(self.name)
        precision = 0
        recall = 0

        tp, fp, fn = 0, 0, 0
        count = 1
        s_iou, nb = 0, 0
        for key in keys:
            img_path = os.path.join(
                self.args.data_path, "images\\train2017", key)
            img_path = os.path.normpath(img_path)

            image = cv2.imread(img_path)
            image = np.array(image, dtype=np.float)
            image = np.uint8(image)
            original_img = image.copy()
            image_width, image_height = image.shape[1], image.shape[0]
            width_scale, height_scale = input_size / \
                image.shape[1], input_size / image.shape[0]

            image = cv2.resize(image, (input_size, input_size))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = preprocess_fn(image)
            image = np.expand_dims(image, axis=0)

            labels = np.array(validation_dictionary[key])
            gt_boxes = np.array(labels[:, :4], dtype=np.float)
            # if len(gt_boxes) > 3:
            #     continue

            # prediction and prediction decoding
            y_pred = self.ssd.predict(x=image)
            y_pred = DecodeSSDPredictions(img_height=input_size,
                                          img_width=input_size,
                                          name="Decoded Predictions",
                                          num_predictions=200,
                                          confidence_threshold=self.args.confidence_threshold)(y_pred)
            y_pred = np.array(y_pred)
            y_pred = np.squeeze(y_pred, axis=0)

            # Anchor Box Decoding
            for pred in y_pred:
                pred[2] = max(int(pred[2] / width_scale), 1)
                pred[3] = max(int(pred[3] / height_scale), 1)
                pred[4] = min(int(pred[4] / width_scale), image_width - 1)
                pred[5] = min(int(pred[5] / height_scale), image_height - 1)

            y_pred = y_pred[:, 2:6] if len(y_pred) != 0 else []

            y_pred = np.array(y_pred)

            # Ground Truth Decoding
            gt_boxes[:, 2] += gt_boxes[:, 0]
            gt_boxes[:, 3] += gt_boxes[:, 1]
            for bbox in gt_boxes:
                bbox[0] = max(int(bbox[0]), 1)
                bbox[1] = max(int(bbox[1]), 1)
                bbox[2] = min(int(bbox[2]), image_width - 1)
                bbox[3] = min(int(bbox[3]), image_height - 1)

            gt_flags = np.zeros(gt_boxes.shape[0])
            pred_flags = np.zeros(y_pred.shape[0])
            
            for gt_index, gt_box in enumerate(gt_boxes):
                flag = False
                maximum = 0
                for pred_index, pred_box in enumerate(y_pred):
                    iou = utils.iou(np.expand_dims(gt_box, axis=0), np.expand_dims(pred_box, axis=0))
                    if iou > 0.5 and flag is False:
                        flag = True
                    else:
                        pred_flags[pred_index] += 1

                    if iou > maximum:
                        maximum = iou
                
                s_iou += maximum
                if flag is True:
                    gt_flags[gt_index] = 1 #flag for true positive predictions| One TP per GT box

            pred_flags = pred_flags == len(gt_boxes)
            tp += np.sum(gt_flags)
            fp += np.sum(pred_flags)
            fn += (len(gt_boxes) - np.sum(gt_flags))
            nb += len(gt_boxes)

            # print("{}/{}".format(count, size))
            count += 1
            # self.evaluate(image=img_path)

        # m_iou = iou / nb
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        print(f" == confidence_threshold: {self.args.confidence_threshold}")
        # print(f" == mIoU: {m_iou}")
        print(f" == recall: {recall}")
        print(f" == precision: {precision}")


if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    parser = ssd_parser()
    args = parser.parse_args()

    with open(os.path.normpath(
        os.path.join(os.getcwd(), args.config, "configs.json"),
    ), 'rb') as reader:
        config = json.load(reader)
    ssd_loss = Loss()
    # model = tf.keras.models.load_model(filepath = os.path.join(args.save_dir, "ssd_vgg16model_test.h5"),
    #                                                             custom_objects={'DefaultBoxes' : DefaultBoxes,
    #                                                                             'L2Normalization' : L2Normalization,
    #                                                                             'Loss' : ssd_loss.compute})
    ssd = SSD(args=args, config=config)
    # print(ssd.ssd.summary())
    # tf.keras.utils.plot_model(ssd.ssd, to_file="model.png")
    # for layer in ssd.ssd.layers:
    #     print("{}: {}".format(layer.name, layer.trainable))

    # if args.restore_weights:
    #     ssd.restore_weights()
    #     if args.evaluate is True:
    #         if args.image_file is None:
    #             ssd.evaluate_test()
    #         else:
    #             ssd.evaluate(image_file=args.image_file)

    # if args.train:
    #     ssd.train()
    ssd.restore_weights()
    # ssd.train()
    # # # ssd.evaluate(image_file='000000425226.jpg')
    ssd.evaluate_test()