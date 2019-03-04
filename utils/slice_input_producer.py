#!/usr/bin/env python
# -*- coding:utf-8 -*-

# import tensorflow as tf
# import numpy as np
#
# # 样本个数
# sample_num = 5
# # 设置迭代次数
# epoch_num = 2
# # 设置一个批次中包含样本个数
# batch_size = 3
# # 计算每一轮epoch中含有的batch个数
# batch_total = int(sample_num / batch_size) + 1
#
#
# # 生成4个数据和标签
# def generate_data(sample_num=sample_num):
#     labels = np.asarray(range(0, sample_num))
#     images = np.random.random([sample_num, 224, 224, 3])
#     print('image size {},label size :{}'.format(images.shape, labels.shape))
#
#     return images, labels
#
#
# def get_batch_data(batch_size=batch_size):
#     images, label = generate_data()
#     # 数据类型转换为tf.float32
#     images = tf.cast(images, tf.float32)
#     label = tf.cast(label, tf.int32)
#
#     # 从tensor列表中按顺序或随机抽取一个tensor
#     input_queue = tf.train.slice_input_producer([images, label], shuffle=False)
#
#     image_batch, label_batch = tf.train.batch(input_queue, batch_size=batch_size, num_threads=1, capacity=64)
#     return image_batch, label_batch
#
#
# image_batch, label_batch = get_batch_data(batch_size=batch_size)
#
# with tf.Session() as sess:
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(sess, coord)
#     try:
#         for i in range(epoch_num):  # 每一轮迭代
#             print
#             '************'
#             for j in range(batch_total):  # 每一个batch
#                 print
#                 '--------'
#                 # 获取每一个batch中batch_size个样本和标签
#                 image_batch_v, label_batch_v = sess.run([image_batch, label_batch])
#                 # for k in
#                 print(image_batch_v.shape, label_batch_v)
#     except tf.errors.OutOfRangeError:
#         print("done")
#     finally:
#         coord.request_stop()
#     coord.join(threads)


# 用于通过读取图片的path,然后解码成图片数组的形式，最后返回batch个图片数组
import glob
import tensorflow as tf
import matplotlib.pyplot as plt

path_list = r'/media/rjw/Ran-software/dataset/iqa_dataset/LIVE/jp2k/'
img_path = glob.glob(path_list + '*.bmp')
label=[i for i in range(len(img_path))]
img_path = tf.convert_to_tensor(img_path, dtype=tf.string)

# 这里img_path,不放在数组里面
# num_epochs = 1,表示将文件下所有的图片都使用一次
# num_epochs和tf.train.slice_input_producer()中是一样的
# 此参数可以用来设置训练的 epochs
image = tf.train.slice_input_producer([img_path,label], num_epochs=1,shuffle=False)


# load one image and decode img
def load_img(path_queue):
    # 创建一个队列读取器，然后解码成数组
    #    reader = tf.WholeFileReader()
    #    key,value = reader.read(path_queue)
    file_contents = tf.read_file(path_queue[0])
    img = tf.image.decode_bmp(file_contents, channels=3)
    # 这里很有必要，否则会出错
    # 感觉这个地方貌似只能解码3通道以上的图片
    img = tf.image.resize_images(img, size=(100, 100))
    # img = tf.reshape(img,shape=(50,50,4))
    return img,path_queue[1]


img,label = load_img(image)
print(img.shape)
image_batch ,label_batch= tf.train.batch([img,label], batch_size=20)

with tf.Session() as sess:
    # initializer for num_epochs
    tf.local_variables_initializer().run()
    coord = tf.train.Coordinator()
    thread = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:
        while not coord.should_stop():
            imgs,labels = sess.run([image_batch ,label_batch])
            print(imgs.shape,labels)
    except tf.errors.OutOfRangeError:
        print('done')
    finally:
        coord.request_stop()
    coord.join(thread)
