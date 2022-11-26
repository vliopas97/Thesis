import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import utils


def one_hot_class_label(classname, label_maps):
    """Turn classname to one hot encoded label.

    Args:
        - classname: String representing the classname
        - label_maps: A list of strings containing all the classes

    Returns:
        - A numpy array of shape ( len(label_maps, )

    Raises:
        - Classname not included in label maps
    """

    assert classname in label_maps, "classname must be included in label maps"
    temp = np.zeros((len(label_maps)), dtype=np.int)
    temp[label_maps.index(classname)] = 1
    return temp


def read_sample(image_path, label_path):
    image_path = image_path.strip("\n")
    label_path = label_path.strip("\n")
    assert os.path.exists(image_path), "Image file does not exist."
    assert os.path.exists(label_path), "Label file does not exist."

    image = cv2.imread(image_path)
    polygons, classes = [], []
    xml_root = ET.parse(label_path).getroot()
    objects = xml_root.findall("object")
    for obj in objects:
        name = obj.find("name").text
        polygon = obj.find("polygon")
        x1 = float(polygon.find("x1").text)
        y1 = float(polygon.find("y1").text)
        x2 = float(polygon.find("x2").text)
        y2 = float(polygon.find("y2").text)
        x3 = float(polygon.find("x3").text)
        y3 = float(polygon.find("y3").text)
        x4 = float(polygon.find("x4").text)
        y4 = float(polygon.find("y4").text)
        polygons.append([x1, y1, x2, y2, x3, y3, x4, y4])
        classes.append(name)

    polygons = np.array(polygons)
    polygons = np.reshape(polygons, (polygons.shape[0], 4, 2))

    return np.array(image, dtype=np.float), np.array(polygons, dtype=np.float), classes


def match_boxes(
        gt_boxes,
        anchors,
        match_threshold,
        neutral_threshold):
    """ Matches ground truth bounding boxes to anchors based on the SSD paper.
    'We begin by matching each ground truth box to the anchor with the best jaccard overlap (as in MultiBox [7]).
    Unlike MultiBox, we then match anchors to any ground truth with jaccard overlap higher than a threshold (0.5)'
    Args:
        - gt_boxes: A numpy array or tensor of shape (num_gt_boxes, 4). Structure [cx, cy, w, h]
        - default_boxes: A numpy array of tensor of shape (num_default_boxes, 4). Structure [cx, cy, w, h]
        - threshold: A float representing a target to decide whether the box is matched
        - neutral_threshodl: A numpy array of tensor of shape (num_default_boxes, 4). Structure [cx, cy, w, h]
    Returns:
        - matches: A numpy array of shape (num_matches, 2). The first index in the last dimension is the index
          of the ground truth box and the last index is the anchor index.
        - neutral_boxes: A numpy array of shape (num_neutral_boxes, 2). The first index in the last dimension is the index
          of the ground truth box and the last index is the anchor index.
    Raises:
        - Either the shape of ground truth's boxes array or the anchors array is not 2
    """

    assert len(gt_boxes.shape) == 2, "Shape of gt_boxes must be 2"
    assert len(anchors.shape) == 2, "Shape of anchors must be 2"

    # convert boxes coords to xmin, ymin, xmax, ymax
    gt_boxes = utils.centroid2minmax(gt_boxes)
    anchors = utils.centroid2minmax(anchors)

    n_boxes = gt_boxes.shape[0]
    n_anchors = anchors.shape[0]

    matches = np.zeros((n_boxes, 2), dtype=np.int)

    for i in range(n_boxes):
        gt_box = gt_boxes[i]
        gt_box = np.tile(np.expand_dims(gt_box, axis=0), (n_anchors, 1))
        ious = utils.iou(gt_box, anchors)
        matches[i] = [i, np.argmax(ious)]

    gt_boxes = np.tile(np.expand_dims(gt_boxes, axis=1), (1, n_anchors, 1))
    anchors = np.tile(np.expand_dims(anchors, axis=0), (n_boxes, 1, 1))
    ious = utils.iou(gt_boxes, anchors)
    ious[:, matches[:, 1]] = 0

    # get iou scores between gt and anchors
    matched_gt_indices = np.argmax(ious, axis=0)
    matched_ious = ious[matched_gt_indices, list(range(n_anchors))]
    matched_anchors_indices = np.nonzero(matched_ious >= match_threshold)[0]
    matched_gt_indices = matched_gt_indices[matched_anchors_indices]

    matches = np.concatenate(
        [matches,
         np.concatenate(
             [
                 np.expand_dims(matched_gt_indices, axis=-1),
                 np.expand_dims(
                     matched_anchors_indices, axis=-1)
             ],
             axis=-1)
         ], axis=0
    )
    ious[:, matches[:, 1]] = 0

    # find neutral boxes (ious that are neutral_threshold < iou < threshold)
    background_gt_indices = np.argmax(ious, axis=0)
    background_gt_ious = ious[background_gt_indices, list(range(n_anchors))]
    neutral_anchors_indices = np.nonzero(
        background_gt_ious >= neutral_threshold)[0]
    neutral_gt_indices = background_gt_indices[neutral_anchors_indices]

    neutral_boxes = np.concatenate([
        np.expand_dims(neutral_gt_indices, axis=-1),
        np.expand_dims(neutral_anchors_indices, axis=-1)],
        axis=-1)

    return matches, neutral_boxes


def encode_boxes(y):
    """Encode the label to a proper format suitable for training SSD network

    Args:
        - y : A numpy of shape (n_boxes, n_classes + 12) representing a label sample

    Returns:
        - A numpy array with the same shape as y but its boxes values have been encoded to the proper SSD format

    """
    epsilon = 10e-5
    
    gt_boxes = y[:, -12: -8]
    df_boxes = y[:, -8: -4]
    variances = y[:, -4:]
    encoded_gt_boxes_cx = ((gt_boxes[:, 0] - df_boxes[:, 0]) / (df_boxes[:, 2])) / np.sqrt(variances[:, 0])
    encoded_gt_boxes_cy = ((gt_boxes[:, 1] - df_boxes[:, 1]) / (df_boxes[:, 3])) / np.sqrt(variances[:, 1])
    encoded_gt_boxes_w = np.log(epsilon + gt_boxes[:, 2] / df_boxes[:, 2]) / np.sqrt(variances[:, 2])
    encoded_gt_boxes_h = np.log(epsilon + gt_boxes[:, 3] / df_boxes[:, 3]) / np.sqrt(variances[:, 3])
    y[:, -12] = encoded_gt_boxes_cx
    y[:, -11] = encoded_gt_boxes_cy
    y[:, -10] = encoded_gt_boxes_w
    y[:, -9] = encoded_gt_boxes_h
    return y