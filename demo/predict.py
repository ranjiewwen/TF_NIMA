#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
created predict.py by rjw at 19-1-11 in WHU.
"""

import os
import tensorflow as tf
from net.VGG16_model import vgg16,fully_connection
from metrics.srocc import mean_score
import matplotlib.pyplot as plt

model_checkpoint_path="../experiments/tid2013/nima/model.ckpt-8000"

image = tf.placeholder(tf.float32,[None,224,224,3])
vgg = vgg16(image)
x = fully_connection(vgg.pool5, 128, 1.0)
scores_hat = tf.nn.softmax(x)
saver = tf.train.Saver()


filename = "img/i16_23_5.bmp"
# image_raw_data = tf.gfile.FastGFile(filename, 'r').read()
image_raw_data = tf.read_file(filename)


with tf.Session() as sess:
    saver.restore(sess, model_checkpoint_path)

    img = tf.image.decode_bmp(image_raw_data, channels=3)
    img1 = tf.image.resize_images(img, [224, 224])
    img2 = tf.image.per_image_standardization(img1)  # 将图片标准化
    img3 = tf.expand_dims(img2, dim=0)

    input = sess.run(img3)
    # input = img3.eval()
    scores_p = sess.run(scores_hat,feed_dict={image:input})
    mean = mean_score(scores_p)

    print(mean)
    plt.figure(filename)
    plt.imshow(img.eval())
    plt.title("predict score mean is :{}".format(mean))
    plt.axis("off")
    plt.show()


