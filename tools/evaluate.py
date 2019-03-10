#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
created evaluate.py by rjw at 19-1-10 in WHU.
"""
import argparse
import os

import tensorflow as tf

from data.image_reader_nima import ImageReader
from loss.EMD_loss import _emd
from loss.reg_loss import reg_l2
from metrics.srocc import scores_stats, mean_score, evaluate_metric
from net.VGG16_model import vgg16, fully_connection
from utils.logger import setup_logger

BASE_PATH = '/media/rjw/Ran-software/dataset/iqa_dataset'


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


# specifying default parameters
def process_command_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Tensorflow Image Enhancement DPED Dataset Training")

    ## Path related arguments
    parser.add_argument('--exp_name', type=str, default="nima", help='experiment name')
    parser.add_argument('--data_dir', type=str, default=BASE_PATH, help='the root path of dataset')
    parser.add_argument('--train_list', type=str, default='tid2013_train.txt', help='data list for read image.')
    parser.add_argument('--test_list', type=str, default='tid2013_test.txt', help='data list for read image.')
    parser.add_argument('--ckpt_dir', type=str, default=os.path.abspath('..') + '/experiments',
                        help='the path of ckpt file')
    parser.add_argument('--logs_dir', type=str, default=os.path.abspath('..') + '/experiments',
                        help='the path of tensorboard logs')

    ## dataset related arguments
    parser.add_argument('--dataset', default='tid2013', type=str, choices=["LIVE", "CSIQ", "tid2013"],
                        help='datset choice')
    parser.add_argument('--crop_width', type=int, default=224, help='train patch width')
    parser.add_argument('--crop_height', type=int, default=224, help='train patch height')

    ## train related arguments
    parser.add_argument('--is_training', type=str2bool, default=False, help='whether to train or test.')
    parser.add_argument('--is_eval', type=str2bool, default=True, help='whether to test.')

    args = parser.parse_args()
    return args


def evaluate(args):
    graph = tf.Graph()

    with graph.as_default() as g:
        with tf.name_scope("create_test_inputs"):
            test_reader = ImageReader(args.data_dir,
                                      args.test_list,
                                      (args.crop_height, args.crop_width),
                                      False,
                                      )
            test_image, test_score, test_mean_std = test_reader.image, test_reader.score, test_reader.mean_std
            test_image, test_score, test_mean_std = tf.expand_dims(test_image, dim=0), tf.expand_dims(test_score,
                                                                                                      dim=0), tf.expand_dims(
                test_mean_std, dim=0)  # Add one batch dimension.

        # # placeholders for training data
        imgs = tf.placeholder(tf.float32, [None, args.crop_height, args.crop_width, 3])
        scores = tf.placeholder(tf.float32, [None, 10])
        with tf.name_scope("create_models"):
            vgg = vgg16(imgs)
            x = fully_connection(vgg.pool5, 128, 1.0)
            scores_hat = tf.nn.softmax(x)

        means = tf.placeholder(tf.float32, [None, 1])
        with tf.name_scope("create_loss"):
            emd_loss = _emd(scores, scores_hat)
            mean_ = scores_stats(scores_hat)
            l2_loss = reg_l2(means, mean_)
            loss = emd_loss + l2_loss * 0.0

        saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=5)

    with tf.Session(graph=graph) as sess:

        sess.run(tf.global_variables_initializer())

        ckpt = tf.train.get_checkpoint_state(args.ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            logger.info('loading checkpoint:' + ckpt.model_checkpoint_path)  # model.ckpt-8000
            saver.restore(sess, ckpt.model_checkpoint_path)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        score_set = []
        label_set = []
        for i in range(len(test_reader.image_list)):
            image_, score_, mean_std_ = sess.run([test_image, test_score, test_mean_std])
            label_set.append(mean_std_[:, 0])
            mean_std_ = mean_std_[:, 0].reshape(-1, 1)
            loss_, scores_hat_ = sess.run([loss, scores_hat],
                                          feed_dict={imgs: image_, scores: score_, means: mean_std_})
            # mean,std=scores_stats(scores_hat_)
            mean = mean_score(scores_hat_)
            score_set.append(mean)
            if i % 50 == 0:
                logger.info("test image:{}/{}, true_mean_mos/predict_mos is {}/{},the emd loss: {}.".format(i, len(
                    test_reader.image_list), mean_std_[:, 0], mean, loss_))
                logger.info("image score_:{}".format(score_))
                logger.info("image score_hat:{}".format(scores_hat_))

        srocc, krocc, plcc, rmse, mse = evaluate_metric(label_set, score_set)
        logger.info(
            "SROCC_v: %.3f\t KROCC: %.3f\t PLCC_v: %.3f\t RMSE_v: %.3f\t mse: %.3f\n" % (srocc, krocc, plcc, rmse, mse))

        logger.info("Test finish!")
        coord.request_stop()
        coord.join(threads)


def main():
    args = process_command_args()

    if args.dataset == 'tid2013':
        args.train_list = 'tid2013_train.txt'
        args.test_list = 'tid2013_test.txt'
    elif args.dataset == 'live':
        args.train_list = 'live_train.txt'
        args.test_list = 'live_test.txt'
    elif args.dataset == 'csiq':
        args.train_list = 'csiq_train.txt'
        args.test_list = 'csiq_test.txt'
    else:
        logger.info("datasets is not in LIVE, CSIQ, tid2013")

    output_dir = os.path.join(args.ckpt_dir, args.dataset)
    args.data_dir = os.path.join(args.data_dir, args.dataset)
    args.ckpt_dir = os.path.join(args.ckpt_dir, args.dataset, args.exp_name)
    args.logs_dir = os.path.join(args.logs_dir, args.dataset, "logs")

    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)

    global logger
    logger = setup_logger("TF_NIMA_evaluating_"+args.dataset, output_dir, "test_")
    logger.info(args)

    evaluate(args)


if __name__ == "__main__":
    main()
