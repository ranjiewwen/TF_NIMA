#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
created inceptionv2.py by rjw at 19-3-9 in WHU.
"""

import tensorflow as tf
slim = tf.contrib.slim
from tensorflow.contrib.slim.nets import inception


def get_model(images, num_classes=10, is_training=True, weight_decay=4e-5,
              dropout_keep_prob=0.75):
    """Neural Image Assessment from https://arxiv.org/abs/1709.05424

    Talebi, Hossein, and Peyman Milanfar. "NIMA: Neural Image Assessment."
    arXiv preprint arXiv:1709.05424 (2017).

    Args:
      images: a tensor of shape [batch_size, height, width, channels].
      num_classes: number of predicted classes. Defaults to 10.
      is_training: whether is training or not.
      weight_decay: the weight decay to use for regularizing the model.
      dropout_keep_prob: the percentage of activation values that are retained.
        Defaults to 0.75

    Returns:
      predictions: a tensor of size [batch_size, num_classes].
      end_points: a dictionary from components of the network.
    """
    arg_scope = inception.inception_v2_arg_scope(weight_decay=weight_decay)
    with slim.arg_scope(arg_scope):
        logits, end_points = inception.inception_v2(
            images, num_classes, is_training=is_training,
            dropout_keep_prob=dropout_keep_prob)

    predictions = tf.nn.softmax(logits)

    return predictions, end_points