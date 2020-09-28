# author: kcgarikipati@gmail.com
"""contains tf utilities"""

import tensorflow as tf


def prediction(logits):
    """ Calculates softmax probability and mask values from pixel logits
    Args:
        logits (4-D Tensor): (N, H, W, 2)
    Returns:
        Predicted pixel-wise probabilities (4-D Tensor): (N, H, W)
        Predicted mask (4-D Tensor): (N, H, W, 1)
        confidence score (1-D Tensor): ()
    """
    # softmax activation
    probs = tf.nn.softmax(logits)
    _, h, w, _ = probs.get_shape()

    masks = tf.reshape(probs, [-1, 2])

    # confidence score for all pixels
    # conf = tf.reduce_mean(tf.reduce_max(masks, axis=1))

    # mask = 1 if class=1 has maximum probability (i.e. >=0.5)
    masks = tf.argmax(masks, axis=1)
    masks = tf.cast(masks, dtype=tf.float32)
    masks = tf.reshape(masks, [-1, h.value, w.value, 1])

    # take one of the class probability from the 2-class distribution
    probs = tf.slice(probs, [0, 0, 0, 1], [-1, -1, -1, 1])

    # limit conf to mask pixels
    temp = tf.multiply(probs, masks)
    conf = tf.reduce_mean(tf.reduce_sum(temp, axis=[1,2,3])/tf.reduce_sum(masks, axis=[1,2,3]))
    conf = tf.cond(tf.is_nan(conf), lambda: tf.constant(0.0), lambda: conf)

    probs = tf.squeeze(probs, axis=-1)

    return probs, masks, conf


def encoded_logits(y_true):
    """ Converts mask to one-hot encoded logits
        Args:
            y_true (4-D Tensor): (N, H, W, 1)
        Returns:
            one-hot encoded logits 43-D Tensor): (N, H, W, 2)
        """
    logits = tf.one_hot(indices=tf.squeeze(tf.cast(y_true, dtype=tf.int32), [-1]), depth=2,
                             on_value=1.0, off_value=0.0, axis= -1)
    return logits