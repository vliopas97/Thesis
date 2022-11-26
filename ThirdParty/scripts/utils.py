import numpy as np
import tensorflow as tf


def get_number_anchor_boxes(aspect_ratios):
    """ Get the number of default boxes for each grid cell based on the number of aspect ratios
    and whether to add a extra box for aspect ratio 1

    Args:
    - aspect_ratios: A list containing the different aspect ratios of default boxes.

    Returns:
    - An integer for the number of default boxes.
    """
    return len(aspect_ratios) + 1


def generate_anchor_boxes_for_feature_map(
    feature_map_shape,
    image_shape,
    scale,
    next_scale,
    aspect_ratios,
    variances
):
    """ Generates a 4D Tensor representing default boxes.
    Note:
    - The structure of a default box is [xmin, ymin, xmax, ymax]
    Args:
    - feature_map_shape: The shape of the feature map. (must be square)
    - image_shape: The shape of the input image. (must be square)
    - offset: The offset for the center of the default boxes. The order is (offset_x, offset_y)
    - scale: The current scale of the default boxes.
    - next_scale: The next scale of the default boxes.
    - aspect_ratios: A list of aspect ratios representing the default boxes.
    - variance: ...
    Returns:
    - A 4D numpy array of shape (feature_map_size, feature_map_size, num_anchor_boxes, 8)

    """
    num_anchor_boxes = get_number_anchor_boxes(aspect_ratios)
    
    image_height, image_width, _ = image_shape
    _, feature_height, feature_width, _ = feature_map_shape

    grid_height = image_height / feature_height
    grid_width = image_width / feature_width
    
    # get all width and height of default boxes
    width_height = []
    # image_size = min(image_height, image_width)
    for ar in aspect_ratios:
        width_height.append([
            image_width * scale * np.sqrt(ar),
            image_height * scale * (1 / np.sqrt(ar))
        ])
        if ar == 1.0: #extra anchor box of the following dimensions for aspect_ratio = 1
            width_height.append([
                image_width * np.sqrt(scale * next_scale) * np.sqrt(ar),
                image_height * np.sqrt(scale * next_scale) * (1 / np.sqrt(ar))
            ])
    # print(scale)
    # print(next_scale)
    # for ar in aspect_ratios:
    #     if (ar == 1):
    #         width_height.append([scale * image_size, scale * image_size])
    #         width_height.append([np.sqrt(scale * next_scale) *image_size, np.sqrt(scale * next_scale) *image_size])
    #     else:
    #         width_height.append([scale * image_size * np.sqrt(ar), scale * image_size / np.sqrt(ar)])
            
    width_height = np.array(width_height, dtype=np.float)
    # get all center points of each grid cells
    start = grid_width * 0.5
    end = (feature_width - 0.5) * grid_width
    cx = np.linspace(start, end, feature_width)

    start = grid_height * 0.5
    end = (feature_height - 0.5) * grid_height
    cy = np.linspace(start, end, feature_height)

    cx_grid, cy_grid = np.meshgrid(cx, cy)
    cx_grid, cy_grid = np.expand_dims(
        cx_grid, axis=-1), np.expand_dims(cy_grid, axis=-1)
    cx_grid, cy_grid = np.tile(cx_grid, (1, 1, num_anchor_boxes)), np.tile(
        cy_grid, (1, 1, num_anchor_boxes))
    #
    anchor_boxes = np.zeros(
        (feature_height, feature_width, num_anchor_boxes, 4))
    anchor_boxes[:, :, :, 0] = cx_grid
    anchor_boxes[:, :, :, 1] = cy_grid
    anchor_boxes[:, :, :, 2] = width_height[:, 0]
    anchor_boxes[:, :, :, 3] = width_height[:, 1]

    anchor_boxes[:, :, :, [0, 2]] /= image_width
    anchor_boxes[:, :, :, [1, 3]] /= image_height
    #
    variances_tensor = np.zeros_like(anchor_boxes)
    variances_tensor += variances
    anchor_boxes = np.concatenate([anchor_boxes, variances_tensor], axis=-1)
    # print(anchor_boxes)
    return anchor_boxes


def centroid2minmax(boxes):
    """Centroid to minmax format 
    (cx, cy, w, h) to (xmin, ymin, xmax, ymax)
    Arguments:
        boxes (tensor): Batch of boxes in centroid format
    Returns:
        minmax (tensor): Batch of boxes in minmax format
    """
    minmax = boxes.copy()
    minmax[..., 0] = boxes[..., 0] - (0.5 * boxes[..., 2])
    minmax[..., 1] = boxes[..., 1] - (0.5 * boxes[..., 3])
    minmax[..., 2] = boxes[..., 0] + (0.5 * boxes[..., 2])
    minmax[..., 3] = boxes[..., 1] + (0.5 * boxes[..., 3])
    return minmax


def minmax2centroid(boxes):
    """Minmax to centroid format
    (xmin, xmax, ymin, ymax) to (cx, cy, w, h)
    Arguments:
        boxes (tensor): Batch of boxes in minmax format
    Returns:
        centroid (tensor): Batch of boxes in centroid format
    """
    centroid = np.copy(boxes).astype(np.float)
    width = np.abs(boxes[..., 0] - boxes[...,2])
    height = np.abs(boxes[..., 1] - boxes[...,3])
    centroid[0] = boxes[..., 0] + 0.5 * width
    centroid[1] = boxes[..., 1] + 0.5 * height
    centroid[2] = width
    centroid[3] = height
    return centroid


def iou(boxes1, boxes2):
    """Compute IoU of batch boxes1 and boxes2
    Arguments:
        boxes1 (tensor): Boxes coordinates in pixels
        boxes2 (tensor): Boxes coordinates in pixels
    Returns:
        iou (tensor): intersectiin of union of areas of
            boxes1 and boxes2
    Warning:
        boxes have to be in corners format (xmin, ymin, xmax, ymax)
    """
    # if boxes1.ndim > 2: raise ValueError("boxes1 must have rank either 1 or 2, but has rank {}.".format(boxes1.ndim))
    # if boxes2.ndim > 2: raise ValueError("boxes2 must have rank either 1 or 2, but has rank {}.".format(boxes2.ndim))

    if boxes1.ndim == 1: boxes1 = np.expand_dims(boxes1, axis=0)
    if boxes2.ndim == 1: boxes2 = np.expand_dims(boxes2, axis=0)

    if not (boxes1.shape[-1] == boxes2.shape[-1] == 4): raise ValueError("All boxes must consist of 4 coordinates, but the boxes in `boxes1` and `boxes2` have {} and {} coordinates, respectively.".format(boxes1.shape[1], boxes2.shape[1]))


    # assert boxes1.shape == boxes2.shape

    xmin_intersect = np.maximum(boxes1[..., 0], boxes2[..., 0])
    ymin_intersect = np.maximum(boxes1[..., 1], boxes2[..., 1])
    xmax_intersect = np.minimum(boxes1[..., 2], boxes2[..., 2])
    ymax_intersect = np.minimum(boxes1[..., 3], boxes2[..., 3])

    intersect = (xmax_intersect - xmin_intersect) * (ymax_intersect - ymin_intersect)
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    union = area1 + area2 - intersect
    iou = intersect / union

    iou[xmax_intersect < xmin_intersect] = 0
    iou[ymax_intersect < ymin_intersect] = 0
    iou[iou < 0] = 0
    iou[iou > 1] = 0

    return iou
