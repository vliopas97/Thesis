"""Label utility functions
Main use: labeling, dictionary of colors,
label retrieval, loading label csv file,
drawing label on an image
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import csv
import json
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from random import randint

from deep_sort.tracker import Tracker
from deep_sort.detection import Detection


class LabelUtil():
    """Utility class for labeling"""

    def __init__(self, config, width_scale, height_scale):
        self.config = config
        self.width_scale = width_scale
        self.height_scale = height_scale

    def get_box_color(self, index=None):
        """Retrieve plt-compatible color string based on object index"""
        colors = ['w', 'r', 'b', 'g', 'c', 'm', 'y', 'g', 'c', 'm', 'k']
        if index is None:
            return colors[randint(0, len(colors) - 1)]
        return colors[index % len(colors)]

    def get_box_rgbcolor(self, index=None):
        """Retrieve rgv color based on object index"""
        colors = [(0, 0, 0), (255, 0, 0), (0, 0, 255),
                  (0, 255, 0), (128, 128, 0)]
        if index is None:
            return colors[randint(0, len(colors) - 1)]
        return colors[index % len(colors)]

    def index2class(self, index=0):
        """Convert index(int) to class name (string)"""
        classes = self.config["classes"]
        return classes[index]

    def class2index(self, class_="background"):
        """Convert a class name(string) to an index(int)"""
        classes = self.config["classes"]
        return classes.index(class_)

    def show_labels__(self, image, labels, ax=None):
        """Draw bounding box on an object given box coords"""
        if ax is None:
            _, ax = plt.subplots(1)
            ax.imshow(image)
        for label in labels:
            category = int(self.class2index(label[-1]))
            color = self.get_box_color(category)
            rect = Rectangle(label[:2],
                             label[2],
                             label[3],
                             linewidth=1,
                             edgecolor=color,
                             facecolor='none')
            ax.add_patch(rect)
        plt.show()

    def show_labels(self, image, bbox, threshold, print_box=False):
        #color = self.get_box_rgbcolor(self.class2index(bbox[0]))
        color = self.get_box_rgbcolor(int(bbox[0]))
        classname = self.index2class(int(bbox[0]))
        confidence_score = bbox[1]

        image_height, image_width, _ = image.shape

        score = f"{'%.2f' % (confidence_score * 100)}%"
        if(confidence_score > threshold):
            if(print_box):
                print(" -- {} {} {} {} {} {}".format(bbox[0], score, bbox[2], bbox[3], bbox[4], bbox[5]))

        if (confidence_score <= 1 and confidence_score >= threshold):
            xmin = max(int(bbox[2] / self.width_scale), 1)
            ymin = max(int(bbox[3] / self.height_scale), 1)
            xmax = min(int(bbox[4] / self.width_scale), image_width - 1)
            ymax = min(int(bbox[5] / self.height_scale), image_height - 1)

            cv2.putText(
                image,
                classname.upper(),
                (int(xmin), int(ymin)),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                color=color
            )

            cv2.rectangle(
                image,
                (xmin, ymin),
                (xmax, ymax),
                color=color,
                thickness=2
            )

    def show_tracks(self, image, track, print_box=False):
        image_height, image_width, _ = image.shape

        xmin, ymin, xmax, ymax = track.to_tlbr().astype(np.int)
        
        if(print_box):
            print(" -- {} {} {} {}".format(xmin, ymin, xmax, ymax))
            
        xmin = max(int(xmin / self.width_scale), 1)
        ymin = max(int(ymin / self.height_scale), 1)
        xmax = min(int(xmax / self.width_scale), image_width - 1)
        ymax = min(int(ymax / self.height_scale), image_height - 1)

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
        cv2.putText(image, str(track.track_id), (xmin, ymin), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)  



def get_label_dictionary(filepath, labels=[]):
    with open(filepath, 'rb') as reader:
        file = json.load(reader)

    classNames = {}
    for cat in file['categories']:
        if (cat['name'] in labels[1:]):
            classNames[cat['id']] = cat['name']

    dictionary = {}
    for annotation in file['annotations']:
        if annotation['category_id'] in classNames:
            key = "{:012d}".format(annotation['image_id']) + '.jpg'
            # key = annotation['image_id'] +'.jpg'
            label = [*annotation['bbox'],
                     classNames[annotation['category_id']]]
            if key in dictionary:
                dictionary[key].append(label)
            else:
                dictionary[key] = [label]

    return dictionary
