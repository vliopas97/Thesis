import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import MaxPool2D, Conv2D, Reshape, Concatenate, Activation
from tensorflow.keras.layers import Layer, Input, ZeroPadding2D, Lambda
from tensorflow.keras.layers import ELU, BatchNormalization, MaxPooling2D, ReLU
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import VGG16, vgg16, MobileNetV2
from tensorflow.keras.applications import mobilenet_v2
from tensorflow.keras.optimizers import SGD, Adam
import utils
from layer_utils import DefaultBoxes, DecodeSSDPredictions
import l2_normalization
import tensorflow as tf

# tf.config.optimizer.set_jit(True)
# tf.keras.mixed_precision.experimental.set_policy('mixed_float16')


def get_optimizer(config, args):
    model_config = config["model"]
    if model_config["name"] == "ssd_vgg16":
        return SGD(learning_rate=args.learning_rate,
                   momentum=0.9,
                   decay=0.0005,
                   clipnorm=1,
                   nesterov=False)
    elif model_config["name"] == "ssd300":
        return SGD(learning_rate=args.learning_rate,
                    momentum=0.9,
                    decay=0.0005,
                    clipnorm=1,
                    nesterov=False)                   
    elif model_config["name"] == "ssd_mobilenet":
        return SGD(learning_rate=args.learning_rate,
                    momentum=0.9,
                    decay=0.0005,
                    clipnorm=1,
                    nesterov=False)
    else:
        return Adam(learning_rate=args.learning_rate,
                    beta_1=0.9,
                    beta_2=0.999,
                    decay=0.0)


def get_preprocess_fn(name):
    if name == "ssd_vgg16":
        return vgg16.preprocess_input
    elif name == "ssd_mobilenet":
        return mobilenet_v2.preprocess_input
    else:
        return None


def SSD_VGG16(config, label_maps):

    model_config = config["model"]
    input_shape = (model_config["input_height"], model_config["input_width"], 3)
    n_classes = len(label_maps)
    l2_reg = model_config["l2_regularization"]
    kernel_initializer = model_config["kernel_initializer"]
    default_boxes_config = model_config["default_boxes"]

    swap_channels = [2, 1, 0]
    def input_channel_swap(tensor):
        return tf.keras.backend.stack([tensor[...,swap_channels[0]], tensor[...,swap_channels[1]], tensor[...,swap_channels[2]]], axis=-1)

    input_tensor = Input(shape=input_shape)
    # input_tensor = Lambda(input_channel_swap, output_shape=input_shape, name="BGR2RGB")(input_tensor)
    input_tensor = ZeroPadding2D(padding=(2, 2))(input_tensor)

    base = VGG16(input_tensor=input_tensor, classes=n_classes,
                 weights='imagenet', include_top=False)
    
    base = Model(inputs=base.input,
                 outputs=base.get_layer('block5_conv3').output)

    base.get_layer("input_1")._name = "input"
    for layer in base.layers:
        if "pool" in layer.name:
            new_name = layer.name.replace("block", "")
            new_name = new_name.split("_")
            new_name = f"{new_name[1]}{new_name[0]}"
        else:
            new_name = layer.name.replace("conv", "")
            new_name = new_name.replace("block", "conv")
        base.get_layer(layer.name)._name = new_name
        base.get_layer(layer.name)._kernel_initializer = "he_normal"
        base.get_layer(layer.name)._kernel_regularizer = l2(l2_reg)
        layer.trainable = False  # each layer of the base network should not be trainable

    def conv_block_1(input, filters, name, padding='valid', dilation_rate=(1, 1), strides=(1, 1)):
        return Conv2D(
            filters,
            kernel_size=(3, 3),
            strides=strides,
            activation='relu',
            padding=padding,
            dilation_rate=dilation_rate,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=l2(l2_reg),
            name=name)(input)

    def conv_block_2(input, filters, name, padding='valid', dilation_rate=(1, 1), strides=(1, 1)):
        return Conv2D(
            filters,
            kernel_size=(1, 1),
            strides=strides,
            activation='relu',
            padding=padding,
            dilation_rate=dilation_rate,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=l2(l2_reg),
            name=name)(input)

    pool5 = MaxPool2D(
        pool_size=(3, 3),
        strides=(1, 1),
        padding="same",
        name="pool5")(base.get_layer('conv5_3').output)

    fc6 = conv_block_1(input=pool5, filters=1024, padding="same",
                       dilation_rate=(6, 6), name="fc6")
    fc7 = conv_block_2(input=fc6, filters=1024, padding="same", name="fc7")
    conv8_1 = conv_block_2(input=fc7, filters=256,
                           padding="valid", name="conv8_1")
    conv8_2 = conv_block_1(input=conv8_1, filters=512,
                           padding="same", strides=(2, 2), name="conv8_2")
    conv9_1 = conv_block_2(input=conv8_2, filters=128,
                           padding="valid", name="conv9_1")
    conv9_2 = conv_block_1(input=conv9_1, filters=256,
                           padding="same", strides=(2, 2), name="conv9_2")
    conv10_1 = conv_block_2(input=conv9_2, filters=128,
                            padding="valid", name="conv10_1")
    conv10_2 = conv_block_1(input=conv10_1, filters=256,
                            padding="valid", name="conv10_2")
    conv11_1 = conv_block_2(input=conv10_2, filters=128,
                            padding="valid", name="conv11_1")
    conv11_2 = conv_block_1(input=conv11_1, filters=256,
                            padding="valid", name="conv11_2")

    model = Model(inputs=base.input, outputs=conv11_2)

    # constructing prediction layers for classes and offsets

    scales = np.linspace(
        default_boxes_config["min_scale"],
        default_boxes_config["max_scale"],
        len(default_boxes_config["layers"]))

    classes = []
    boxes = []
    anchors = []

    for i, layer in enumerate(default_boxes_config["layers"]):
        num_default_boxes = utils.get_number_anchor_boxes(
            layer["aspect_ratios"])
        x = model.get_layer(layer["name"]).output
        layer_name = layer["name"]

        if layer_name == "conv4_3":
            layer_name = f"{layer_name}_norm"
            x = l2_normalization.L2Normalization(
                gamma_init=20, name=layer_name, dtype='float32')(x)

        classes_ = Conv2D(
            filters=num_default_boxes * n_classes,
            kernel_size=(3, 3),
            padding='same',
            kernel_initializer=kernel_initializer,
            kernel_regularizer=l2(l2_reg),
            name=f"{layer_name}_classes_internal")(x)
        classes_ = Reshape(
            (-1, n_classes), name=f"{layer_name}_classes")(classes_)
        boxes_ = Conv2D(
            filters=num_default_boxes * 4,
            kernel_size=(3, 3),
            padding='same',
            kernel_initializer=kernel_initializer,
            kernel_regularizer=l2(l2_reg),
            name=f"{layer_name}_boxes_internal")(x)
        boxes_ = Reshape((-1, 4), name=f"{layer_name}_boxes")(boxes_)
        anchors_ = DefaultBoxes(
            image_shape=input_shape,
            scale=scales[i],
            next_scale=scales[i+1] if i +
            1 <= len(default_boxes_config["layers"]) - 1 else 1,
            aspect_ratios=layer["aspect_ratios"],
            variances=default_boxes_config["variances"],
            name=f"{layer_name}_anchors_internal")(x)
        anchors_ = Reshape((-1, 8), name=f"{layer_name}_anchors")(anchors_)
        classes.append(classes_)
        boxes.append(boxes_)
        anchors.append(anchors_)

    classes = Concatenate(axis=-2)(classes)
    classes = Activation('softmax', name="classes")(classes)

    boxes = Concatenate(axis=-2)(boxes)

    anchors = Concatenate(axis=-2, name="anchors")(anchors)

    output = Concatenate(axis=-1, dtype='float32')([classes, boxes, anchors])
    # if is_training:
    #    return Model(inputs=base.input, outputs = output)

    # if inference:
    #     output = DecodeSSDPredictions(
    #     img_height=model_config["input_height"],
    #     img_width=model_config["input_width"],
    #     num_predictions=10,
    #     name="decoded_outputs")(output)

    return Model(inputs=base.input, outputs=output)

def SSD7(config,
        label_maps):

    model_config = config["model"]
    input_height, input_width = model_config["input_height"], model_config["input_width"]
    n_classes = len(label_maps)
    l2_reg = l2(model_config["l2_regularization"])
    kernel_initializer = model_config["kernel_initializer"]
    default_boxes_config = model_config["default_boxes"]
    subtract_mean = 127.5
    divide_by_stddev = 127.5

    input_img = Input(shape=(input_height, input_width, 3))

    def identity_layer(tensor):
        return tensor

    def input_mean_normalization(tensor):
        return tensor - np.array(subtract_mean)

    def input_stddev_normalization(tensor):
        return tensor / np.array(divide_by_stddev)

    x = Lambda(identity_layer, output_shape=(input_height, input_width, 3), name="identity_layer")(input_img)
    x = Lambda(input_mean_normalization, output_shape=(input_height, input_width, 3), name="mean_normalization")(x)
    x = Lambda(input_stddev_normalization, output_shape=(input_height, input_width, 3), name="stddev_normalization")(x)


    # Backbone Network
    conv1 = Conv2D(32, (5, 5), strides=(1, 1), padding='same', kernel_initializer=kernel_initializer, kernel_regularizer=l2_reg, name='conv1')(x)
    conv1 = BatchNormalization(axis=3, momentum=0.99, name='bn1')(conv1)
    conv1 = ELU(name='elu1')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), name='pool1')(conv1)

    conv2 = Conv2D(48, (3, 3), strides=(1, 1), padding='same', kernel_initializer=kernel_initializer,kernel_regularizer=l2_reg, name='conv2')(pool1)
    conv2 = BatchNormalization(axis=3, momentum=0.99, name='bn2')(conv2)
    conv2 = ELU(name='elu2')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), name='pool2')(conv2)

    conv3 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer=kernel_initializer, kernel_regularizer=l2_reg, name='conv3')(pool2)
    conv3 = BatchNormalization(axis=3, momentum=0.99, name='bn3')(conv3)
    conv3 = ELU(name='elu3')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), name='pool3')(conv3)

    conv4 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer=kernel_initializer, kernel_regularizer=l2_reg, name='conv4')(pool3)
    conv4 = BatchNormalization(axis=3, momentum=0.99, name='bn4')(conv4)
    conv4 = ELU(name='elu4')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2), name='pool4')(conv4)

    conv5 = Conv2D(48, (3, 3), strides=(1, 1), padding='same', kernel_initializer=kernel_initializer, kernel_regularizer=l2_reg, name='conv5')(pool4)
    conv5 = BatchNormalization(axis=3, momentum=0.99, name='bn5')(conv5)
    conv5 = ELU(name='elu5')(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2), name='pool5')(conv5)

    conv6 = Conv2D(48, (3, 3), strides=(1, 1), padding='same', kernel_initializer=kernel_initializer, kernel_regularizer=l2_reg, name='conv6')(pool5)
    conv6 = BatchNormalization(axis=3, momentum=0.99, name='bn6')(conv6)
    conv6 = ELU(name='elu6')(conv6)
    pool6 = MaxPooling2D(pool_size=(2, 2), name='pool6')(conv6)

    conv7 = Conv2D(32, (3, 3), strides=(1, 1), padding='same', kernel_initializer=kernel_initializer, kernel_regularizer=l2_reg, name='conv7')(pool6)
    conv7 = BatchNormalization(axis=3, momentum=0.99, name='bn7')(conv7)
    conv7 = ELU(name='elu7')(conv7)

    scales = default_boxes_config["scales"]
    aspect_ratios = default_boxes_config["aspect_ratios"]

    conv = [conv4, conv5, conv6, conv7]
    classes = []
    boxes = []
    anchors = []
    feature_shapes = []

    for i in range(4, 8):
        n_boxes = utils.get_number_anchor_boxes(aspect_ratios[i-4])

        classes_ = Conv2D(n_boxes * n_classes, (3, 3), strides=(1, 1), padding='same', kernel_initializer=kernel_initializer, kernel_regularizer=l2_reg, name = 'classes{}'.format(i))(conv[i - 4])
        classes_ = Reshape((-1, n_classes), name = 'classes{}_reshape'.format(i))(classes_)

        boxes_ = Conv2D(n_boxes * 4, (3, 3), strides=(1, 1), padding='same', kernel_initializer=kernel_initializer, kernel_regularizer=l2_reg, name='boxes{}'.format(i))(conv[i - 4])
        feature_shapes.append(np.array(boxes_.shape))
        boxes_ = Reshape((-1, 4), name= 'boxes{}_reshape'.format(i))(boxes_)

        anchors_ = DefaultBoxes(image_shape=(input_height, input_width, 3),
        scale=scales[i - 4],
        next_scale=scales[i + 1 - 4],
        aspect_ratios=aspect_ratios[i-4],
        variances=[0.1, 0.1, 0.2, 0.2],
        name='anchors{}'.format(i))(conv[i-4])
        anchors_ = Reshape((-1, 8), name= 'anchors{}_reshape'.format(i))(anchors_)

        classes.append(classes_)
        boxes.append(boxes_)
        anchors.append(anchors_)

    classes = Concatenate(axis=-2)(classes)
    classes = Activation('softmax', name="classes")(classes)

    boxes = Concatenate(axis=-2)(boxes)

    anchors = Concatenate(axis=-2, name="anchors")(anchors)

    output = Concatenate(axis=-1)([classes, boxes, anchors])
    return Model(inputs=input_img, outputs=output), feature_shapes

def SSD_Mobilenet(config, label_maps):
    model_config = config["model"]
    input_height, input_width = model_config["input_height"], model_config["input_width"]
    n_classes = len(label_maps)

    l2_reg = model_config["l2_regularization"]
    kernel_initializer = model_config["kernel_initializer"]
    default_boxes_config = model_config["default_boxes"]

    input_shape = (input_height, input_width, 3)

    base = MobileNetV2(input_shape=input_shape,
                        alpha=config["model"]["width_multiplier"],
                        classes=n_classes,
                        weights='imagenet',
                        include_top=False)

    base = Model(inputs=base.input, outputs=base.get_layer('block_16_project_BN').output)
    base.get_layer("input_1")._name = "input"
    for layer in base.layers:
        base.get_layer(layer.name)._kernel_initializer = "he-normal"
        base.get_layer(layer.name)._kernel_regularizer = l2(l2_reg)
        layer.trainable = False

    conv_16 = base.get_layer("block_16_project_BN").output

    def conv_block_1(x, filters, name):
        x = Conv2D(
            filters=filters,
            kernel_size=(1, 1),
            padding="valid",
            kernel_initializer=kernel_initializer,
            kernel_regularizer=l2(l2_reg),
            name=name,
            use_bias=False)(x)
        x = BatchNormalization(name=f"{name}/bn")(x)
        x = ReLU(name=f"{name}/relu")(x)
        return x

    def conv_block_2(x, filters, name):
        x = Conv2D(
            filters=filters,
            kernel_size=(3, 3),
            padding="same",
            kernel_initializer=kernel_initializer,
            kernel_regularizer=l2(l2_reg),
            name=name,
            use_bias=False,
            strides=(2, 2))(x)
        x = BatchNormalization(name=f"{name}/bn")(x)
        x = ReLU(name=f"{name}/relu")(x)
        return x

    conv17_1 = conv_block_1(x=conv_16, filters=256, name="conv17_1")
    conv17_2 = conv_block_2(x=conv17_1, filters=512, name="conv17_2")
    conv18_1 = conv_block_1(x=conv17_2, filters=128, name="conv18_1")
    conv18_2 = conv_block_2(x=conv18_1, filters=256, name="conv18_2")
    conv19_1 = conv_block_1(x=conv18_2, filters=128, name="conv19_1")
    conv19_2 = conv_block_2(x=conv19_1, filters=256, name="conv19_2")
    conv20_1 = conv_block_1(x=conv19_2, filters=128, name="conv20_1")
    conv20_2 = conv_block_2(x=conv20_1, filters=256, name="conv20_2")

    model = Model(inputs=base.input, outputs=conv20_2)

    scales = np.linspace(
        default_boxes_config["min_scale"],
        default_boxes_config["max_scale"],
        len(default_boxes_config["layers"])
    )    

    classes = []
    boxes = []
    anchors = []

    for i, layer in enumerate(default_boxes_config["layers"]):
        num_default_boxes = utils.get_number_anchor_boxes(
            layer["aspect_ratios"])
        x = model.get_layer(layer["name"]).output
        layer_name = layer["name"]

        classes_ = Conv2D(
            filters=num_default_boxes * n_classes,
            kernel_size=(3, 3),
            padding='same',
            kernel_initializer=kernel_initializer,
            kernel_regularizer=l2(l2_reg),
            name=f"{layer_name}_classes_internal")(x)
        classes_ = Reshape(
            (-1, n_classes), name=f"{layer_name}_classes")(classes_)
        boxes_ = Conv2D(
            filters=num_default_boxes * 4,
            kernel_size=(3, 3),
            padding='same',
            kernel_initializer=kernel_initializer,
            kernel_regularizer=l2(l2_reg),
            name=f"{layer_name}_boxes_internal")(x)
        boxes_ = Reshape((-1, 4), name=f"{layer_name}_boxes")(boxes_)
        anchors_ = DefaultBoxes(
            image_shape=input_shape,
            scale=scales[i],
            next_scale=scales[i+1] if i +
            1 <= len(default_boxes_config["layers"]) - 1 else 1,
            aspect_ratios=layer["aspect_ratios"],
            variances=default_boxes_config["variances"],
            name=f"{layer_name}_anchors_internal")(x)
        anchors_ = Reshape((-1, 8), name=f"{layer_name}_anchors")(anchors_)
        classes.append(classes_)
        boxes.append(boxes_)
        anchors.append(anchors_)   

    classes = Concatenate(axis=-2)(classes)
    classes = Activation('softmax', name="classes")(classes)

    boxes = Concatenate(axis=-2)(boxes)

    anchors = Concatenate(axis=-2, name="anchors")(anchors)

    output = Concatenate(axis=-1)([classes, boxes, anchors])

    # if is_training:
    #    return Model(inputs=base.input, outputs = output)

    # decoded_output = DecodeSSDPredictions(
    #    input_size = model_config["input_size"],
    #    num_predictions=num_predictions,
    #    name="decoded_outputs")(output)

    return Model(inputs=base.input, outputs=output)
    
def SSD300(config, label_maps):
    model_config = config["model"]
    input_shape = (model_config["input_height"], model_config["input_width"], 3)
    n_classes = len(label_maps)
    l2_reg = model_config["l2_regularization"]
    kernel_initializer = model_config["kernel_initializer"]
    default_boxes_config = model_config["default_boxes"]
    scales = default_boxes_config["scales"]
    aspect_ratios = default_boxes_config["aspect_ratios"]

    def identity_layer(tensor):
        return tensor

    subtract_mean=[123, 117, 104]

    def input_mean_normalization(tensor):
        return tensor - np.array(subtract_mean)

    swap_channels = [2, 1, 0]
    def input_channel_swap(tensor):
        return tf.keras.backend.stack([tensor[...,swap_channels[0]], tensor[...,swap_channels[1]], tensor[...,swap_channels[2]]], axis=-1)


    x = Input(shape=input_shape)
    x_ = Lambda(identity_layer, output_shape=input_shape, name='identity_layer')(x)
    x_ = Lambda(input_mean_normalization, output_shape=input_shape, name='mean_normalization')(x_)
    x1 = Lambda(input_channel_swap, output_shape=input_shape, name= 'input_channel_swap')(x_)

    conv1_1 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv1_1')(x1)
    conv1_2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv1_2')(conv1_1)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool1')(conv1_2)

    conv2_1 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv2_1')(pool1)
    conv2_2 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv2_2')(conv2_1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool2')(conv2_2)

    conv3_1 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv3_1')(pool2)
    conv3_2 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv3_2')(conv3_1)
    conv3_3 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv3_3')(conv3_2)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool3')(conv3_3)

    conv4_1 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv4_1')(pool3)
    conv4_2 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv4_2')(conv4_1)
    conv4_3 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv4_3')(conv4_2)
    pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool4')(conv4_3)

    conv5_1 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv5_1')(pool4)
    conv5_2 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv5_2')(conv5_1)
    conv5_3 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv5_3')(conv5_2)
    pool5 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='pool5')(conv5_3)

    fc6 = Conv2D(1024, (3, 3), dilation_rate=(6, 6), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='fc6')(pool5)

    fc7 = Conv2D(1024, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='fc7')(fc6)

    conv6_1 = Conv2D(256, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv6_1')(fc7)
    conv6_1 = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv6_padding')(conv6_1)
    conv6_2 = Conv2D(512, (3, 3), strides=(2, 2), activation='relu', padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv6_2')(conv6_1)

    conv7_1 = Conv2D(128, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv7_1')(conv6_2)
    conv7_1 = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv7_padding')(conv7_1)
    conv7_2 = Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv7_2')(conv7_1)

    conv8_1 = Conv2D(128, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv8_1')(conv7_2)
    conv8_2 = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv8_2')(conv8_1)

    conv9_1 = Conv2D(128, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv9_1')(conv8_2)
    conv9_2 = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv9_2')(conv9_1)

    # Feed conv4_3 into the L2 normalization layer
    conv4_3_norm = l2_normalization.L2Normalization(gamma_init=20, name='conv4_3_norm')(conv4_3)

    # conv = [conv4_3_norm, fc7, conv6_2, conv7_2, conv8_2, conv9_2]

    # classes = []
    # boxes = []
    # anchors = []
    # for i in range(len(conv)):
    #     n_boxes = utils.get_number_anchor_boxes(aspect_ratios=aspect_ratios[i])

    #     classes_ = Conv2D(n_boxes * n_classes, (3, 3), strides=(1, 1), padding='same', kernel_initializer=kernel_initializer, kernel_regularizer=l2(l2_reg), name = 'classes{}'.format(i))(conv[i])
    #     classes_ = Reshape((-1, n_classes), name='classes{}_reshape'.format(i))(classes_)

    #     boxes_ = Conv2D(n_boxes * 4, (3, 3), strides=(1, 1), padding='same', kernel_initializer=kernel_initializer, kernel_regularizer=l2(l2_reg), name='boxes{}'.format(i))(conv[i])
    #     boxes_ = Reshape((-1, 4), name= 'boxes{}_reshape'.format(i))(boxes_)

    #     anchors_ = DefaultBoxes(image_shape=input_shape,
    #     scale=scales[i],
    #     next_scale=scales[i + 1 ],
    #     aspect_ratios=aspect_ratios[i],
    #     variances=[0.1, 0.1, 0.2, 0.2],
    #     name='anchors{}'.format(i))(conv[i])
    #     anchors_ = Reshape((-1, 8), name= 'anchors{}_reshape'.format(i))(anchors_) 

    #     classes.append(classes_)
    #     boxes.append(boxes_)
    #     anchors.append(anchors_)

    

    # classes = Concatenate(axis=-2)(classes)
    # classes = Activation('softmax', name="classes")(classes)

    # boxes = Concatenate(axis=-2)(boxes)

    # anchors = Concatenate(axis=-2, name="anchors")(anchors)

    # output = Concatenate(axis=-1)([classes, boxes, anchors])
    # return Model(inputs=x, outputs=output)
    # 
    # 
    # We precidt `n_classes` confidence values for each box, hence the confidence predictors have depth `n_boxes * n_classes`
    # Output shape of the confidence layers: `(batch, height, width, n_boxes * n_classes)`
    n_boxes=[]
    n_boxes.append(utils.get_number_anchor_boxes(aspect_ratios[0]))
    n_boxes.append(utils.get_number_anchor_boxes(aspect_ratios[1]))
    n_boxes.append(utils.get_number_anchor_boxes(aspect_ratios[2]))
    n_boxes.append(utils.get_number_anchor_boxes(aspect_ratios[3]))
    n_boxes.append(utils.get_number_anchor_boxes(aspect_ratios[4]))
    n_boxes.append(utils.get_number_anchor_boxes(aspect_ratios[5]))
    conv4_3_norm_mbox_conf = Conv2D(n_boxes[0] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv4_3_norm_mbox_conf')(conv4_3_norm)
    fc7_mbox_conf = Conv2D(n_boxes[1] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='fc7_mbox_conf')(fc7)
    conv6_2_mbox_conf = Conv2D(n_boxes[2] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv6_2_mbox_conf')(conv6_2)
    conv7_2_mbox_conf = Conv2D(n_boxes[3] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv7_2_mbox_conf')(conv7_2)
    conv8_2_mbox_conf = Conv2D(n_boxes[4] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv8_2_mbox_conf')(conv8_2)
    conv9_2_mbox_conf = Conv2D(n_boxes[5] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv9_2_mbox_conf')(conv9_2)
    # We predict 4 box coordinates for each box, hence the localization predictors have depth `n_boxes * 4`
    # Output shape of the localization layers: `(batch, height, width, n_boxes * 4)`
    conv4_3_norm_mbox_loc = Conv2D(n_boxes[0] * 4, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv4_3_norm_mbox_loc')(conv4_3_norm)
    fc7_mbox_loc = Conv2D(n_boxes[1] * 4, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='fc7_mbox_loc')(fc7)
    conv6_2_mbox_loc = Conv2D(n_boxes[2] * 4, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv6_2_mbox_loc')(conv6_2)
    conv7_2_mbox_loc = Conv2D(n_boxes[3] * 4, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv7_2_mbox_loc')(conv7_2)
    conv8_2_mbox_loc = Conv2D(n_boxes[4] * 4, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv8_2_mbox_loc')(conv8_2)
    conv9_2_mbox_loc = Conv2D(n_boxes[5] * 4, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv9_2_mbox_loc')(conv9_2)

        # Output shape of anchors: `(batch, height, width, n_boxes, 8)`
    conv4_3_norm_mbox_priorbox = DefaultBoxes(image_shape=input_shape,
                                            scale=scales[0],
                                            next_scale=scales[1],
                                            aspect_ratios=aspect_ratios[0],
                                            variances=[0.1, 0.1, 0.2, 0.2],
                                            name='conv4_3_norm_mbox_priorbox')(conv4_3_norm_mbox_loc)
    fc7_mbox_priorbox = DefaultBoxes(image_shape=input_shape,
                                            scale=scales[1],
                                            next_scale=scales[2],
                                            aspect_ratios=aspect_ratios[1],
                                            variances=[0.1, 0.1, 0.2, 0.2],
                                            name='fc7_mbox_priorbox')(fc7_mbox_loc)
    conv6_2_mbox_priorbox = DefaultBoxes(image_shape=input_shape,
                                            scale=scales[2],
                                            next_scale=scales[3],
                                            aspect_ratios=aspect_ratios[2],
                                            variances=[0.1, 0.1, 0.2, 0.2],
                                            name='conv6_2_mbox_priorbox')(conv6_2_mbox_loc)
    conv7_2_mbox_priorbox = DefaultBoxes(image_shape=input_shape,
                                            scale=scales[3],
                                            next_scale=scales[4],
                                            aspect_ratios=aspect_ratios[3],
                                            variances=[0.1, 0.1, 0.2, 0.2],
                                            name='conv7_2_mbox_priorbox')(conv7_2_mbox_loc)
    conv8_2_mbox_priorbox = DefaultBoxes(image_shape=input_shape,
                                            scale=scales[4],
                                            next_scale=scales[5],
                                            aspect_ratios=aspect_ratios[4],
                                            variances=[0.1, 0.1, 0.2, 0.2],
                                            name='conv8_2_mbox_priorbox')(conv8_2_mbox_loc)
    conv9_2_mbox_priorbox = DefaultBoxes(image_shape=input_shape,
                                            scale=scales[5],
                                            next_scale=scales[6],
                                            aspect_ratios=aspect_ratios[5],
                                            variances=[0.1, 0.1, 0.2, 0.2],
                                            name='conv9_2_mbox_priorbox')(conv9_2_mbox_loc)
        
    # Reshape the class predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, n_classes)`
    # We want the classes isolated in the last axis to perform softmax on them
    conv4_3_norm_mbox_conf_reshape = Reshape((-1, n_classes), name='conv4_3_norm_mbox_conf_reshape')(conv4_3_norm_mbox_conf)
    fc7_mbox_conf_reshape = Reshape((-1, n_classes), name='fc7_mbox_conf_reshape')(fc7_mbox_conf)
    conv6_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv6_2_mbox_conf_reshape')(conv6_2_mbox_conf)
    conv7_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv7_2_mbox_conf_reshape')(conv7_2_mbox_conf)
    conv8_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv8_2_mbox_conf_reshape')(conv8_2_mbox_conf)
    conv9_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv9_2_mbox_conf_reshape')(conv9_2_mbox_conf)
    # Reshape the box predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, 4)`
    # We want the four box coordinates isolated in the last axis to compute the smooth L1 loss
    conv4_3_norm_mbox_loc_reshape = Reshape((-1, 4), name='conv4_3_norm_mbox_loc_reshape')(conv4_3_norm_mbox_loc)
    fc7_mbox_loc_reshape = Reshape((-1, 4), name='fc7_mbox_loc_reshape')(fc7_mbox_loc)
    conv6_2_mbox_loc_reshape = Reshape((-1, 4), name='conv6_2_mbox_loc_reshape')(conv6_2_mbox_loc)
    conv7_2_mbox_loc_reshape = Reshape((-1, 4), name='conv7_2_mbox_loc_reshape')(conv7_2_mbox_loc)
    conv8_2_mbox_loc_reshape = Reshape((-1, 4), name='conv8_2_mbox_loc_reshape')(conv8_2_mbox_loc)
    conv9_2_mbox_loc_reshape = Reshape((-1, 4), name='conv9_2_mbox_loc_reshape')(conv9_2_mbox_loc)
    # Reshape the anchor box tensors, yielding 3D tensors of shape `(batch, height * width * n_boxes, 8)`
    conv4_3_norm_mbox_priorbox_reshape = Reshape((-1, 8), name='conv4_3_norm_mbox_priorbox_reshape')(conv4_3_norm_mbox_priorbox)
    fc7_mbox_priorbox_reshape = Reshape((-1, 8), name='fc7_mbox_priorbox_reshape')(fc7_mbox_priorbox)
    conv6_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv6_2_mbox_priorbox_reshape')(conv6_2_mbox_priorbox)
    conv7_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv7_2_mbox_priorbox_reshape')(conv7_2_mbox_priorbox)
    conv8_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv8_2_mbox_priorbox_reshape')(conv8_2_mbox_priorbox)
    conv9_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv9_2_mbox_priorbox_reshape')(conv9_2_mbox_priorbox)

        # Axis 0 (batch) and axis 2 (n_classes or 4, respectively) are identical for all layer predictions,
    # so we want to concatenate along axis 1, the number of boxes per layer
    # Output shape of `mbox_conf`: (batch, n_boxes_total, n_classes)
    mbox_conf = Concatenate(axis=1, name='mbox_conf')([conv4_3_norm_mbox_conf_reshape,
                                                       fc7_mbox_conf_reshape,
                                                       conv6_2_mbox_conf_reshape,
                                                       conv7_2_mbox_conf_reshape,
                                                       conv8_2_mbox_conf_reshape,
                                                       conv9_2_mbox_conf_reshape])

    # Output shape of `mbox_loc`: (batch, n_boxes_total, 4)
    mbox_loc = Concatenate(axis=1, name='mbox_loc')([conv4_3_norm_mbox_loc_reshape,
                                                     fc7_mbox_loc_reshape,
                                                     conv6_2_mbox_loc_reshape,
                                                     conv7_2_mbox_loc_reshape,
                                                     conv8_2_mbox_loc_reshape,
                                                     conv9_2_mbox_loc_reshape])

    # Output shape of `mbox_priorbox`: (batch, n_boxes_total, 8)
    mbox_priorbox = Concatenate(axis=1, name='mbox_priorbox')([conv4_3_norm_mbox_priorbox_reshape,
                                                               fc7_mbox_priorbox_reshape,
                                                               conv6_2_mbox_priorbox_reshape,
                                                               conv7_2_mbox_priorbox_reshape,
                                                               conv8_2_mbox_priorbox_reshape,
                                                               conv9_2_mbox_priorbox_reshape])

    # The box coordinate predictions will go into the loss function just the way they are,
    # but for the class predictions, we'll apply a softmax activation layer first
    mbox_conf_softmax = Activation('softmax', name='mbox_conf_softmax')(mbox_conf)

    # Concatenate the class and box predictions and the anchors to one large predictions vector
    # Output shape of `predictions`: (batch, n_boxes_total, n_classes + 4 + 8)
    predictions = Concatenate(axis=2, name='predictions')([mbox_conf_softmax, mbox_loc, mbox_priorbox])
    model = Model(inputs=x, outputs=predictions)
    return model

