#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
from math import pi, exp, sqrt

import tensorflow as tf
from maxentropy.skmaxent import MinDivergenceModel


# for tid2013 mos-[0-9]
def f(x, mean, var):
    var = var * var
    sigma = 1e-9
    return exp(-(x - mean) ** 2 / (2 * var + sigma)) / sqrt(2 * pi * var + sigma)


def get_score_distribution(mean, var):
    scores_list = []
    for i in range(10):
        scores_list.append(f(i, mean, var))
    return scores_list


def get_distribution(mean, var):
    s = np.random.normal(mean, var, 10000)
    s = np.rint(s)
    a = np.histogram(s, bins=np.arange(1, 12), density=True)
    return a[0]


# the maximised distribution must satisfy the mean for each sample
def get_features():
    def f0(x):
        return x

    return [f0]


def get_max_entropy_distribution(mean):
    SAMPLESPACE = np.arange(10)
    features = get_features()

    model = MinDivergenceModel(features, samplespace=SAMPLESPACE, algorithm='CG')

    # set the desired feature expectations and fit the model
    X = np.array([[mean]])
    model.fit(X)

    return model.probdist()


# reference :https://github.com/DrSleep/tensorflow-deeplab-resnet/blob/master/deeplab_resnet/image_reader.py
def read_labeled_image_list(data_dir, data_list):
    """Reads txt file containing paths to images and ground truth masks.

    Args:
      data_dir: path to the directory with images and masks.
      data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.

    Returns:
      Two lists with all file names for images and masks, respectively.
    """
    file = os.path.join(data_dir, data_list)
    f = open(file, 'r')
    images = []
    scores = []
    mean_std = []
    for line in f:
        try:
            image, dmos, dmos_std = line.strip("\n").split(' ')  # i.split()[0]
        except ValueError:  # Adhoc for test.
            image = dmos = line.strip("\n")
        images.append(os.path.join(data_dir, image))
        mean_std.append((float(dmos), float(dmos_std)))  ## mean_std contains tuple:(dmos,doms_std)

        # score = get_distribution(float(dmos), float(dmos_std))
        score = get_max_entropy_distribution(float(dmos))
        scores.append(score.tolist())

    return images, scores, mean_std


def read_images_from_disk(input_queue, input_size, is_training):  # optional pre-processing arguments
    """Read one image and its corresponding mask with optional pre-processing.

    Args:
      input_queue: tf queue with paths to the image and its mask.
      input_size: a tuple with (height, width) values.
                  If not given, return images of original size.
      img_mean: vector of mean colour values.

    Returns:
      Two tensors: the decoded image and its mask.
    """

    img_contents = tf.read_file(input_queue[0])
    score = input_queue[1]
    mean_std = input_queue[2]

    img = tf.image.decode_bmp(img_contents, channels=3)
    # img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
    # img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)
    # # Extract mean.
    # img -= img_mean

    h, w = input_size
    if is_training:
        # resize image
        img = tf.image.resize_images(img, [256, 256])

        # img = tf.reshape(img, shape=(224, 224, 3)) # bug
        img = tf.random_crop(img, [h, w, 3])
        # img = tf.image.resize_images(img, size=(224, 224))

        img = tf.image.random_flip_left_right(img)

    else:
        img = tf.image.resize_images(img, [h, w])

    img = tf.image.per_image_standardization(img)  # 将图片标准化
    # img = (tf.cast(img, tf.float32) - 127.5) / 127.5
    # score = tf.reshape(score, [10])
    # score = score / tf.reduce_sum(score, axis=-1, keepdims=True)

    return img, score, mean_std


class ImageReader(object):
    '''Generic ImageReader which reads images and corresponding segmentation
       masks from the disk, and enqueues them into a TensorFlow queue.
    '''

    def __init__(self, data_dir, data_list, input_size, is_training):
        '''Initialise an ImageReader.

        Args:
          data_dir: path to the directory with images and masks.
          data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.
          input_size: a tuple with (height, width) values, to which all the images will be resized.
        '''
        self.data_dir = data_dir
        self.data_list = data_list
        self.input_size = input_size

        self.image_list, self.scores_list, self.mean_std_list = read_labeled_image_list(self.data_dir, self.data_list)
        self.images = tf.convert_to_tensor(self.image_list, dtype=tf.string)
        self.scores = tf.convert_to_tensor(self.scores_list, dtype=tf.float32)
        self.mean_stds = tf.convert_to_tensor(self.mean_std_list, dtype=tf.float32)

        self.queue = tf.train.slice_input_producer([self.images, self.scores, self.mean_stds],
                                                   shuffle=is_training)  # not shuffling if it is val
        self.image, self.score, self.mean_std = read_images_from_disk(self.queue, self.input_size, is_training)

    def dequeue(self, num_elements):
        '''Pack images and labels into a batch.

        Args:
          num_elements: the batch size.

        Returns:
          Two tensors of size (batch_size, h, w, {3, 1}) for images and masks.'''
        image_batch, label_batch, mean_std_batch = tf.train.batch([self.image, self.score, self.mean_std],
                                                                  num_elements, capacity=100)
        return image_batch, label_batch, mean_std_batch


import matplotlib.pyplot as plt

# Define DB information
BASE_PATH = '/media/rjw/Ran-software/dataset/iqa_dataset/tid2013/'
import numpy as np

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

if __name__ == "__main__":

    # image_raw_data = tf.gfile.FastGFile('../demo//img001.bmp', 'rb').read()
    # sess=tf.Session()
    # with sess.as_default():
    #     # img_data = tf.image.decode_bmp(image_raw_data)
    #     img_data = tf.image.decode_bmp(image_raw_data, channels=3)
    #     img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img_data)
    #     img_data = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)
    #     print(img_data.eval())
    #     plt.imshow(img_data.eval())
    #     plt.show()

    with tf.Graph().as_default(), tf.Session() as sess:

        # Load reader.
        with tf.name_scope("create_inputs"):
            reader = ImageReader(
                BASE_PATH,
                'tid2013_train.txt',
                [224, 224],
                True
            )
            image_batch, label_batch, mean_std_batch = reader.dequeue(16)

        with tf.name_scope("create_test_inputs"):
            test_reader = ImageReader(BASE_PATH,
                                      'tid2013_test.txt',
                                      [224, 224],
                                      False
                                      )
            test_image, test_score, test_mean_std = test_reader.image, test_reader.score, test_reader.mean_std
            test_image, test_score, test_mean_std = tf.expand_dims(test_image, dim=0), tf.expand_dims(test_score,
                                                                                                      dim=0), tf.expand_dims(
                test_mean_std, dim=0)

        print(len(reader.image_list))  # 2400
        # sess.run(tf.global_variables_initializer())
        tf.local_variables_initializer().run()
        coord = tf.train.Coordinator()
        thred = tf.train.start_queue_runners(sess=sess, coord=coord)

        iters_per_epoch = len(reader.image_list) / 32
        print(iters_per_epoch)

        image_batch_, label_batch_, mean_std_batch_ = sess.run([image_batch, label_batch, mean_std_batch])

        for i in range(len(test_reader.image_list)):
            image_, score_, mean_std_ = sess.run([test_image, test_score, test_mean_std])
            print(image_.shape, score_.shape, mean_std_.shape)

        # # look for source image
        # for i in range(5):
        #     k=sess.run(reader.queue)
        #     print("+++++++++++++++++++++")
        #     print((i,k))
        #     img=Image.open(k[0])
        #     plt.figure('test')
        #     plt.imshow(img)
        #     plt.show()
        #     pass
        try:
            while not coord.should_stop():
                images, labels = sess.run([image_batch, label_batch])
                # import cv2
                # cv2.imshow('test',images[0])
                # cv2.waitKey(0)
                plt.figure('test')
                plt.imshow(images[0])
                # print(images[0])
                plt.show()
                print(images.shape)
                print(labels)
        except tf.errors.OutOfRangeError:
            print("done")
        finally:
            coord.request_stop()
