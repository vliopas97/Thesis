import os
import errno
import argparse
import numpy as np
import cv2
import tensorflow as tf

def _run_in_batches(f, data_dict, out, batch_size):
    data_len = len(out)
    num_batches = int(data_len / batch_size)
    s, e = 0, 0
    for i in range(num_batches):
        s, e = i * batch_size, (i + 1) * batch_size
        batch_data_dict = {k : v[s:e] for k,v in data_dict.items()}
        out[s:e] = f(batch_data_dict)
    if e < len(out):
        batch_data_dict = {k: v[e:] for k, v in data_dict.items()}
        out[e:] = f(batch_data_dict)

def extract_image_patch(image, bbox, patch_shape):
    bbox = np.array(bbox)
    if patch_shape is not None:
        # correct aspect ratio to patch shape
        target_aspect = float(patch_shape[1]) / patch_shape[0]
        new_width = target_aspect * bbox[3]
        bbox[0] -= (new_width - bbox[2]) / 2
        bbox[2] = new_width

    # convert to top left, bottom right
    bbox[2:] += bbox[:2]
    bbox = bbox.astype(np.int)

    # clip at image boundaries
    bbox[:2] = np.maximum(0, bbox[:2])
    bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])

    if ((bbox[0] >= bbox[2]) or (bbox[1] >= bbox[3])):
        return None
    sx, sy, ex, ey = bbox
    image = image[sy:ey, sx:ex]
    image = cv2.resize(image, tuple(patch_shape[::-1]))
    return image


class ImageEncoder(object):

    def __init__(self, checkpoint_filename, input_name="images",
    output_name="features"):
        self.session = tf.compat.v1.Session()
        checkpoint_filename = os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), checkpoint_filename),
        )
        with tf.compat.v1.gfile.GFile(checkpoint_filename, "rb") as file_handle:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(file_handle.read())
        tf.compat.v1.import_graph_def(graph_def)

        self.input_var = tf.compat.v1.get_default_graph().get_tensor_by_name(
            "%s:0" % input_name
        )
        self.output_var = tf.compat.v1.get_default_graph().get_tensor_by_name(
            "%s:0" % output_name
        )
        
        assert len(self.output_var.get_shape()) == 2
        assert len(self.input_var.get_shape()) == 4
        self.feature_dim = self.output_var.get_shape().as_list()[-1]
        self.image_shape = self.input_var.get_shape().as_list()[1:]

    def __call__(self, data_x, batch_size=32):
        out = np.zeros((len(data_x), self.feature_dim), np.float32)
        # self.session.run(self.output_var, feed_dict={self.input_var : data_x})
        _run_in_batches(lambda x : self.session.run(self.output_var, feed_dict=x),
        {self.input_var : data_x}, out, batch_size)
        return out

def create_box_encoder(model_filename, batch_size):
    image_encoder = ImageEncoder(model_filename)
    image_shape = image_encoder.image_shape

    def encoder(image, boxes):
        image_patches = []
        for box in boxes[0]:
            patch = extract_image_patch(image, box, image_shape[:2])
            if patch is None:
                print("WARNING: Failed to extract image patch")
                patch = np.random.uniform(0., 255., image_shape).astype(np.uint8)
            image_patches.append(patch)
        image_patches = np.asarray(image_patches)
        return image_encoder(image_patches, batch_size)

    return encoder

# DEBUG
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.set_logical_device_configuration(gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=1024)])

# encoder = create_box_encoder("C:/Users/Vangelis Liopas/Desktop/deep_sort-master/resources/networks/mars-small128.pb", batch_size=1)
# image = np.ones((1080, 1920, 3), dtype=np.uint8)
# box = np.array([2, 5, 100, 102], dtype=np.float32)
# box = np.expand_dims(box, axis=0)
# box = np.expand_dims(box, axis=0)
# features = encoder(image, box)