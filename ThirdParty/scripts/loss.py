import tensorflow as tf

class Loss:

    def __init__(
            self,
            alpha=1,
            min_negative_boxes=0,
            negative_boxes_ratio=3):
        self.alpha = alpha
        self.min_negative_boxes = min_negative_boxes
        self.negative_boxes_ratio = negative_boxes_ratio

    @staticmethod
    def _smooth_l1(y_true, y_pred):
        abs_loss = tf.abs(y_true - y_pred)
        square_loss = 0.5 * (y_true - y_pred) ** 2
        res = tf.where(tf.less(abs_loss, 1.0), square_loss, abs_loss - 0.5)
        return tf.reduce_sum(res, axis=-1)

    @staticmethod
    def _softmax_loss(y_true, y_pred):
        y_pred = tf.maximum(y_pred, 1e-15)
        return -1 * tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)

    @staticmethod
    def _focal_loss(y_true, y_pred):
        gamma = 2.0
        alpha = 0.25

        y_pred /= tf.keras.backend.sum(y_pred, axis=-1, keepdims=True)

        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.keras.backend.clip(y_pred, epsilon, 1. - epsilon)

        cross_entropy = -y_true * tf.keras.backend.log(y_pred)
        weight = alpha * tf.keras.backend.pow(tf.math.add(1., tf.math.multiply(-1., y_pred)), gamma)
        cross_entropy *= weight

        return tf.keras.backend.sum(cross_entropy, axis=-1)

    def softmax_loss(self, y_true, y_pred):
        # calculate smooth l1 loss and softmax loss for all boxes
        batch_size = tf.shape(y_true)[0]
        num_boxes = tf.shape(y_true)[1]
        bbox_true = y_true[:, :, -12: -8]
        bbox_pred = y_pred[:, :, -12: -8]
        class_true = y_true[:, :, :-12] 
        class_pred = y_pred[:, :, :-12]

        #
        regression_loss = self._smooth_l1(bbox_true, bbox_pred)
        # classification_loss = self._softmax_loss(class_true, class_pred)
        classification_loss = self._focal_loss(class_true, class_pred)
        #
        negatives = class_true[:, :, 0]  # (batch_size, num_boxes)
        positives = tf.reduce_max(class_true[:, :, 1:], axis=-1)  # (batch_size, num_boxes)
        num_positives = tf.cast(tf.reduce_sum(positives), tf.int32)
        #
        pos_regression_loss = tf.reduce_sum(regression_loss * positives, axis=-1)
        pos_classification_loss = tf.reduce_sum(classification_loss * positives, axis=-1)
        #
        neg_classification_loss = classification_loss * negatives
        num_neg_classification_loss = tf.math.count_nonzero(neg_classification_loss, dtype=tf.int32)
        num_neg_classification_loss_keep = tf.minimum(
            tf.maximum(self.negative_boxes_ratio * num_positives, self.min_negative_boxes),
            num_neg_classification_loss
        )

        def f1():
            return tf.zeros([batch_size])

        def f2():
            neg_classification_loss_1d = tf.reshape(neg_classification_loss, [-1])
            _, indices = tf.nn.top_k(
                neg_classification_loss_1d,
                k=num_neg_classification_loss_keep,
                sorted=False
            )
            negatives_keep = tf.scatter_nd(
                indices=tf.expand_dims(indices, axis=1),
                updates=tf.ones_like(indices, dtype=tf.int32),
                shape=tf.shape(neg_classification_loss_1d)
            )
            negatives_keep = tf.cast(tf.reshape(negatives_keep, [batch_size, num_boxes]), tf.float32)
            neg_class_loss = tf.reduce_sum(classification_loss * negatives_keep, axis=-1)
            return neg_class_loss

        neg_classification_loss = tf.cond(tf.equal(num_neg_classification_loss, tf.constant(0)), f1, f2)
        classification_loss = pos_classification_loss + neg_classification_loss

        total = (classification_loss + self.alpha * pos_regression_loss) / tf.maximum(1.0, tf.cast(num_positives, tf.float32))
        total = total * tf.cast(batch_size, tf.float32)
        return total

    def focal_loss(self, y_true, y_pred):
        batch_size = tf.shape(y_true)[0]
        bbox_true = y_true[:, :, -12: -8]
        bbox_pred = y_pred[:, :, -12: -8]
        class_true = y_true[:, :, :-12] 
        class_pred = y_pred[:, :, :-12]

        regression_loss = self._smooth_l1(bbox_true, bbox_pred)
        classification_loss = self._focal_loss(class_true, class_pred)

        positives = tf.reduce_max(class_true[:, :, 1:], axis=-1)
        num_positives = tf.cast(tf.reduce_sum(positives), tf.int32)
        regression_loss = tf.reduce_sum(
        regression_loss * positives, axis=-1)
        classification_loss = tf.reduce_sum(classification_loss, axis=-1)

        # self.alpha = self._getalpha(classification_loss, regression_loss)
        total = (classification_loss + self.alpha * regression_loss) / \
            tf.maximum(1.0, tf.cast(num_positives, tf.float32))
        total = total * tf.cast(batch_size, tf.float32)
        return total
