#author: kcgarikipati@gmail.com

"""contains model class and methods"""

import tensorflow as tf
from utils.tf import prediction
import losses
import sys


def conv_conv_pool(input_, n_filters, kernel_size, training, name, pool=True, batch_norm=True, l2_regularizer=False,
                   dropout=False, seed=None):
    """
    {Conv -> BN -> RELU}x2
    Args:
        input_ (4-D Tensor): (N, H, W, C)
        n_filters (list): number of filters [int, int]
        kernel_size (int): filter kernel size
        training (1-D Tensor): Boolean Tensor
        name (str): name postfix
        pool (bool): If True, MaxPool2D
        batch_norm (bool): normalize weights across batches
        l2_regularizer (bool): regularize conv2d weights
        dropout (bool): dropout after max-pooling
        seed (int): random seed for dropout
    Returns:
        net: output of the Convolution operations
        pool (optional): output of the max pooling operations
    """
    net = input_
    reg_val = 0.1

    with tf.variable_scope("layer_{}".format(name)):
        for i, F in enumerate(n_filters):
            net = tf.layers.conv2d(
                net,
                F, (kernel_size, kernel_size),
                activation=None,
                padding='same',
                kernel_regularizer=tf.contrib.layers.l2_regularizer(reg_val) if l2_regularizer else None,
                name="conv_{}".format(i + 1))
            if batch_norm:
                net = tf.layers.batch_normalization(
                    net, training=training, name="bn_{}".format(i + 1))
            net = tf.nn.relu(net, name="relu{}_{}".format(name, i + 1))

        if pool is False:
            return net
        pooled = tf.layers.max_pooling2d(net, (2, 2), strides=(2, 2), name="pool_{}".format(name))
        if dropout:
            pooled = tf.layers.dropout(pooled, rate=0.5, seed=seed, training=training)
        return net, pooled


def upconv_2D(tensor, n_filter, kernel_size, name, l2_regularizer):
    """
    Up Convolution `tensor` by 2 times
    Args:
        tensor (4-D Tensor): (N, H, W, C)
        n_filter (int): Filter Size
        kernel_size (int): filter kernel size
        name (str): name of upsampling operations
        l2_regularizer (bool): regularize conv2d weights
    Returns:
        output (4-D Tensor): (N, 2 * H, 2 * W, C)
    """
    reg_val = 0.1
    return tf.layers.conv2d_transpose(
            tensor,
            filters=n_filter,
            kernel_size=kernel_size,
            strides=2,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(reg_val) if l2_regularizer else None,
            name="upsample_{}".format(name)
    )


def unet(x, training, init_channels=16, n_layers=4, kernel_size=7, batch_norm=True, gaussian_noise=False,
         l2_regularizer=True, dropout=False, seed = None, dilated=True):
    """
    U-Net model
    Args:
        x (4-D Tensor): Image input of shape (N, H, W, 2)
        training (1-D Tensor): Boolean Tensor
        init_channels (int): initial no. of channels of u-net
        n_layers (int): depth of u-net
        kernel_size (int): conv2d filter kernel size
        batch_norm (bool): use batch normalization
        gaussian_noise (bool): add gaussian noise
        l2_regularizer (bool): regularize weights
        dropout (bool): use dropout
        seed (int): seed for randomization
        dilated (int): use dilated convolutions
    Returns:
        logits (4-D Tensor): (N, H, W, 2)
    """

    # normalize between [-1, 1]
    feed = tf.subtract(x, 0.5)
    feed = tf.multiply(feed, 2.0)

    ch = init_channels
    skip = []

    # encoder block
    for i in range(n_layers):
        conv, feed = conv_conv_pool(feed, [ch, ch], kernel_size, training, name='encoder_{}'.format(i+1), pool=True,
                                    batch_norm=batch_norm, l2_regularizer=l2_regularizer, dropout=dropout, seed=seed)
        skip.append(conv)
        ch = ch*2

    # bottleneck
    feed = conv_conv_pool(feed, [ch, ch], kernel_size, training, name='bottleneck', pool=False,
                          batch_norm=batch_norm, l2_regularizer=l2_regularizer, dropout=dropout, seed=seed)

    if dilated:
        # dilated bottleneck from https://github.com/lyakaap/Kaggle-Carvana-3rd-Place-Solution
        depth = 6
        dilated_layers = []
        for i in range(depth):
            feed = tf.layers.conv2d(feed, ch, (3, 3), padding='same', activation=tf.nn.relu, dilation_rate=2**i)
            dilated_layers.append(feed)

        feed = tf.add_n([feed] + dilated_layers)

    # decoder block
    for i in reversed(range(n_layers)):
        ch = ch//2
        feed = upconv_2D(feed, ch, 2, name='decoder_{}'.format(i+1), l2_regularizer=l2_regularizer)
        concat = tf.concat([feed, skip[i]], axis=-1, name="concat_{}".format(i+1))
        feed = conv_conv_pool(concat, [ch, ch], kernel_size, training, name='decoder_{}'.format(i+1), pool=False,
                              batch_norm=batch_norm, l2_regularizer=l2_regularizer, dropout=dropout, seed=seed)
    # output
    logits = tf.layers.conv2d(feed, 2, (1, 1), padding='same', activation=None, name='logits')
    return logits


class Model:
    def __init__(self, cfg):
        self.cfg = cfg
        self.placeholders = {}

    def create_placeholders(self):
        with tf.name_scope('input'):
            image = tf.placeholder(tf.float32, [None, self.cfg.height, self.cfg.width, 3], name='x')
            mask = tf.placeholder(tf.float32, [None, self.cfg.height, self.cfg.width, 1], name='y')
            weight = tf.placeholder(tf.float32, [None], name='weight')
            train = tf.placeholder(tf.bool, name='is_train')

        self.placeholders = {'image': image, 'weight': weight, 'mask': mask, 'train': train}

    def create_net(self, x, training):
        logits_out = unet(x, training=training, init_channels=self.cfg.init_channels, n_layers=self.cfg.num_layers,
                          kernel_size=self.cfg.ksize, seed=self.cfg.seed,
                          dilated=(True if self.cfg.model == 'unet_dilated' else False))
        return logits_out

    def create_loss_ops(self, logits_pred, weight, y_true):

        # soft means non-thresholded, so takes values in between [0,1]
        y_pred_soft, y_pred, _ = prediction(logits_pred)
        y_pred_soft = tf.expand_dims(y_pred_soft, -1)

        loss_d = losses.dice_loss(y_pred_soft, y_true, weight)
        loss_j = losses.iou_loss(y_pred_soft, y_true, weight)
        loss_p = losses.pixel_wise_loss(logits_pred, y_true, weight)

        if self.cfg.loss == 'dice':
            loss = loss_d
        elif self.cfg.loss == 'jacc':
            loss = loss_j
        elif self.cfg.loss == 'sce':
            loss = loss_p
        else:
            raise ValueError

        # Add summaries.
        tf.summary.scalar("step_losses/dice_loss", loss_d)
        tf.summary.scalar("step_losses/jacc_loss", loss_j)
        tf.summary.scalar("step_losses/cross_entropy_loss", loss_p)

        return [loss, loss_d, loss_p, loss_j]

    def create_train_ops(self, loss):
        # is this the same as using
        #   global_step = tf.Variable(0, name='global_step', trainable=False)
        global_step = tf.train.get_or_create_global_step()

        opt = self.cfg.optimizer
        if opt == 'adam':
            solver = tf.train.AdamOptimizer(self.cfg.learning_rate, beta1=self.cfg.lr_decay)
        elif opt == 'adagrad':
            solver = tf.train.AdagradOptimizer(self.cfg.learning_rate)
        elif opt == 'sgd':
            solver = tf.train.GradientDescentOptimizer(self.cfg.learning_rate)
        elif opt == 'rmsprop':
            solver = tf.train.RMSPropOptimizer(self.cfg.learning_rate)

        # UPDATE_OPS is a collection of ops (operations performed when the graph runs,
        # like multiplication, ReLU, etc.), not variables. Specifically, this collection maintains a
        # list of ops which need to run after every training step.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = solver.minimize(loss, global_step=global_step)

        # for var in tf.trainable_variables():
        #     tf.summary.histogram("parameters/" + var.op.name, var)

        return train_op

    def create_eval_ops(self, logits_pred):

        probs_pred, mask_pred, conf = prediction(logits_pred)
        # assign labels
        with tf.name_scope('output'):
            probs_pred = tf.identity(probs_pred, name='y_pred_soft')
            mask_pred = tf.identity(mask_pred, name='y_pred')
            conf = tf.identity(conf, name='score')

        return [probs_pred, mask_pred, conf]

    def build(self, is_train=True):

        self.create_placeholders()
        X = self.placeholders['image']
        w = self.placeholders['weight']
        y = self.placeholders['mask']
        train = self.placeholders['train']

        logits_pred = self.create_net(X, training=train)
        loss_op = self.create_loss_ops(logits_pred, w, y)
        if is_train:
            train_op = self.create_train_ops(loss_op[0])
        else:
            train_op = tf.no_op()
        eval_op = self.create_eval_ops(logits_pred)
        summary_op = tf.summary.merge_all()

        # Remember the ops we want to run by adding it to a collection.
        # tf.add_to_collection('train_op', train_op)
        # tf.add_to_collection('eval_op', eval_op)
        # tf.add_to_collection('summary_op', summary_op)

        def feed_dict_fn(x_batch, w_batch, y_batch, train_bool):
            return {X: x_batch, w: w_batch, y: y_batch, train: train_bool}

        return loss_op, train_op, eval_op, summary_op, feed_dict_fn
