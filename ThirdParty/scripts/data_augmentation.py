import cv2
import random
import numpy as np
import skimage


def apply_random_brightness(
        image,
        min_delta=-32,
        max_delta=32,
        p=0.5):

    assert min_delta >= -255 and max_delta <= 255.0, "delta must be within range"
    assert p >= 0 and p <= 1, "p must be within range"

    if (random.random() > p):
        return image

    new_img = image.copy()
    d = random.uniform(min_delta, max_delta)
    new_img += d
    new_img = np.clip(new_img, 0, 255)
    return new_img


def apply_random_contrast(
        image,
        min_delta=0.5,
        max_delta=1.5,
        p=0.5):

    assert min_delta >= 0, "min delta can't be negative"
    assert max_delta >= min_delta, "max_delta can't be less than min_delta"
    assert p >= 0 and p <= 1, "p must be within range"

    if (random.random() > p):
        return image

    new_img = image.copy()
    d = random.uniform(min_delta, max_delta)
    new_img *= d
    new_img = np.clip(new_img, 0, 255)
    return new_img


def apply_random_hue(
        image,
        min_delta=-18,
        max_delta=18,
        p=0.5):

    assert min_delta >= -360.0 and max_delta <= 360, "delta must be within range"
    assert p >= 0 and p <= 1, "p must be within range"

    if (random.random() > p):
        return image

    new_img = cv2.cvtColor(np.uint8(image), cv2.COLOR_BGR2HSV)
    new_img = np.array(new_img, dtype=np.float)
    d = random.uniform(min_delta, max_delta)
    new_img[:, :, 0] += d
    new_img = np.clip(new_img, 0, 360)
    new_img = cv2.cvtColor(np.uint8(new_img), cv2.COLOR_HSV2BGR)
    new_img = np.array(new_img, dtype=np.float)
    return new_img


def apply_random_saturation(
        image,
        min_delta=0.5,
        max_delta=1.5,
        p=0.3):

    assert min_delta >= 0, "min delta can't be negative"
    assert max_delta >= min_delta, "max_delta can't be less than min_delta"
    assert p >= 0 and p <= 1, "p must be within range"

    if (random.random() > p):
        return image
    new_img = cv2.cvtColor(np.uint8(image), cv2.COLOR_BGR2HSV)
    new_img = np.array(new_img, dtype=np.float)
    d = random.uniform(min_delta, max_delta)
    new_img[:, :, 1] *= d
    new_img = cv2.cvtColor(np.uint8(new_img), cv2.COLOR_HSV2BGR)
    new_img = np.array(new_img, dtype=np.float)
    return new_img


def apply_random_noise(
        image,
        p=0.5):

    assert p >= 0 and p <= 1, "p must be within range"

    if (random.random() > p):
        return image

    new_img = skimage.util.random_noise(image)
    return new_img


def apply_random_vertical_flip(
        image,
        boxes=None,
        p=0.5):

    assert p >= 0 and p <= 1, "p must be within range"

    if (random.random() > p):
        return image, boxes

    new_boxes = boxes.copy()
    img_center = np.array(image.shape[:2])[::-1]/2
    img_center = np.hstack((img_center, img_center))
    new_boxes[:, [1, 3]] += 2*(img_center[[1, 3]] - new_boxes[:, [1, 3]])
    boxes_height = abs(new_boxes[:, 1] - new_boxes[:, 3])
    new_boxes[:, 1] -= boxes_height
    new_boxes[:, 3] += boxes_height
    return np.array(cv2.flip(np.uint8(image), 0), dtype=np.float), new_boxes


def apply_random_horizontal_flip(
        image,
        boxes=None,
        p=0.5):

    assert p >= 0 and p <= 1, "p must be within range"

    if(random.random() > p):
        return image, boxes

    new_boxes = boxes.copy()
    img_center = np.array(image.shape[:2])[::-1]/2
    img_center = np.hstack((img_center, img_center))
    new_boxes[:, [0, 2]] += 2*(img_center[[0, 2]] - new_boxes[:, [0, 2]])
    boxes_width = abs(new_boxes[:, 0] - new_boxes[:, 2])
    new_boxes[:, 0] -= boxes_width
    new_boxes[:, 2] += boxes_width
    return np.array(cv2.flip(np.uint8(image), 1), dtype=np.float), new_boxes


def apply_random_expand(
        image,
        boxes,
        min_ratio=1,
        max_ratio=4,
        mean=[0.406, 0.456, 0.485],
        p=0.5):

    assert p >= 0, "p must be larger than or equal to zero"
    assert p <= 1, "p must be less than or equal to 1"
    assert min_ratio > 0, "min_ratio must be larger than zero"
    assert max_ratio > min_ratio, "max_ratio must be larger than min_ratio"

    if (random.random() > p):
        return image, boxes

    height, width, depth = image.shape
    ratio = random.uniform(min_ratio, max_ratio)
    left = random.uniform(0, width * ratio - width)
    top = random.uniform(0, height * ratio - height)
    new_img = np.zeros(
        (int(height * ratio), int(width * ratio), depth), dtype=image.dtype)

    new_img[:, :, :] = mean
    new_img[int(top): int(top+height), int(left):int(left+width)] = image
    new_boxes = boxes.copy()
    new_boxes[:, :2] += (int(left), int(top))
    new_boxes[:, 2:] += (int(left), int(top))
    return new_img, new_boxes
