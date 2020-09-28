# author: kcgarikipati@gmail.com
"""contains the various loss functions"""
import tensorflow as tf


def dice_score(y_pred, y_true, weights=None, smooth=1e-7):
    """ Calculates dice score = 2TP/(2TP + FP + FN)
    Args:
        y_true: Tensor of shape (N, H, W, 1)
        y_pred: Tensor of shape (N, H, W, 1)
        weights (1-D Tensor) : (N,) weights
        smooth: Smoothing parameter
    Returns:
        scalar score: dice score between 0 and 1
    """
    y_pred_f = tf.squeeze(y_pred, [-1])
    y_true_f = tf.squeeze(y_true, [-1])
    # calculate score per sample
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=[1,2])
    sum = tf.reduce_sum(y_true_f, axis=[1,2]) + tf.reduce_sum(y_pred_f, axis=[1,2])
    dice_per_image = (2.*intersection + smooth)/(sum + smooth)

    if weights is None:
        return tf.reduce_mean(dice_per_image)
    else:
        weights = tf.reshape(weights, [-1])
        return tf.reduce_sum(dice_per_image * weights) / tf.reduce_sum(weights)


def dice_loss(y_pred, y_true, weights):
    return -dice_score(y_pred, y_true, weights)


def iou_score(y_pred, y_true, weights=None, smooth=1e-7):
    """ Calculates iou/jaccard score = TP/(TP + FP + FN)
        Args:
            y_true: Tensor of shape (N, H, W,1)
            y_pred: Tensor of shape (N, H, W,1)
            weights (1-D Tensor) : (N,) weights
            smooth: Smoothing parameter
        Returns:
            scalar score: iou score between 0 and 1
    """
    # convert to float
    y_pred_f = tf.squeeze(y_pred, [-1])
    y_true_f = tf.squeeze(y_true, [-1])
    # calculate score per single image
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=[1,2])
    sum = tf.reduce_sum(y_true_f, axis=[1,2]) + tf.reduce_sum(y_pred_f, axis=[1,2])
    iou_per_image = (intersection + smooth)/ (sum - intersection + smooth)

    # weigh it across batch
    if weights is None:
        return tf.reduce_mean(iou_per_image)
    else:
        weights = tf.reshape(weights, [-1])
        return tf.reduce_sum(iou_per_image * weights) / tf.reduce_sum(weights)


def iou_loss(y_pred, y_true, weights):
    return -iou_score(y_pred, y_true, weights)


def pixel_wise_loss(logits_pred, y_true, weights=None):
    """Calculates pixel-wise softmax cross entropy loss
       Args:
           logits_pred (4-D Tensor): (N, H, W, 2)
           y_true (3-D Tensor): Image masks of shape (N, H, W, 1)
           weights (1-D Tensor) : (N,) weights
       Returns:
           scalar loss : softmax cross-entropy
    """
    # flatten everything except last dimension
    labels = tf.squeeze(y_true, axis=-1)
    labels = tf.cast(labels, tf.int32)

    # what about tf.nn.softmax_cross_entropy_with_logits_v2 ?
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_pred, labels=labels)
    # loss has shape (NOne, H, W)

    if weights is None:
        # reduce across all dimensions
        return tf.reduce_mean(loss)
    else:
        # multiply pixels of each image in the batch by corresponding weight
        weighted_images = tf.einsum('aij,a->aij', loss, weights)
        weighted_vec = tf.reduce_mean(weighted_images, axis=[1,2]) # per image
        return tf.reduce_sum(weighted_vec) / tf.reduce_sum(weights) # per batch