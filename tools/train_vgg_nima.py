#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
 create by ranjiewen at 20190108 in whu
"""
import argparse
import os
import time
from datetime import datetime

import numpy as np
import tensorflow as tf

from data.image_reader_nima import ImageReader
from loss.EMD_loss import _emd
from loss.reg_loss import reg_l2
from metrics.srocc import scores_stats, mean_score, evaluate_metric
from net.VGG16_model import vgg16, fully_connection
from utils.checkpoint import save
from utils.logger import setup_logger


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


timestamp = datetime.fromtimestamp(time.time()).strftime('%Y%m%d-%H:%M')
experiment_name = os.path.splitext(__file__.split('/')[-1])[0]
BASE_PATH = '/media/rjw/Ran-software/dataset/iqa_dataset'


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
    parser.add_argument('--pretrain_weights', type=str,
                        default=os.path.abspath('..') + "/data/vgg_models/" + 'vgg16_weights.npz')

    ## models retated argumentss
    parser.add_argument('--save_ckpt_file', type=str2bool, default=True,
                        help="whether to save trained checkpoint file ")

    ## dataset related arguments
    parser.add_argument('--dataset', default='tid2013', type=str, choices=["LIVE", "CSIQ", "tid2013"],
                        help='datset choice')
    parser.add_argument('--crop_width', type=int, default=224, help='train patch width')
    parser.add_argument('--crop_height', type=int, default=224, help='train patch height')

    ## train related arguments
    parser.add_argument('--is_training', type=str2bool, default=True, help='whether to train or test.')
    parser.add_argument('--is_eval', type=str2bool, default=True, help='whether to test.')
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--eval_step', type=int, default=500)
    parser.add_argument('--summary_step', type=int, default=2)

    ## optimization related arguments
    parser.add_argument('--learning_rate', type=float, default=3e-5, help='init learning rate')
    parser.add_argument('--iter_max', type=int, default=9000, help='the maxinum of iteration')
    parser.add_argument('--learning_rate_decay_factor', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=4e-5)

    args = parser.parse_args()
    return args


def train(args):
    graph = tf.Graph()
    with graph.as_default():
        global_step = tf.train.create_global_step()
        with tf.name_scope("create_train_inputs"):
            reader = ImageReader(args.data_dir,
                                 args.train_list,
                                 (args.crop_height, args.crop_width),
                                 args.is_training,
                                 )
            image_batch, label_batch, mean_std_batch = reader.dequeue(args.batch_size)

        with tf.name_scope("create_test_inputs"):
            test_reader = ImageReader(args.data_dir,
                                      args.test_list,
                                      (args.crop_height, args.crop_width),
                                      False,
                                      )
            test_image, test_score, test_mean_std = test_reader.image, test_reader.score, test_reader.mean_std
            test_image, test_score, test_mean_std = tf.expand_dims(test_image, dim=0), tf.expand_dims(test_score,
                                                                                                      dim=0), tf.expand_dims(
                test_mean_std, dim=0)
        ## placeholders for training data
        imgs = tf.placeholder(tf.float32, [None, args.crop_height, args.crop_width, 3])
        scores = tf.placeholder(tf.float32, [None, 10])

        with tf.name_scope("create_models"):
            vgg = vgg16(imgs)
            x = fully_connection(vgg.pool5, 128, 0.5)
            scores_hat = tf.nn.softmax(x)

        means = tf.placeholder(tf.float32, [None, 1])
        with tf.name_scope("create_loss"):
            emd_loss_out = _emd(scores, scores_hat)
            mean_hat = scores_stats(scores_hat)
            l2_loss = reg_l2(means, mean_hat)
            loss = emd_loss_out + l2_loss * 0.0

        # decay_steps = len(reader.image_list) / args.batch_size
        # lr = tf.train.exponential_decay(args.learning_rate, global_step, decay_steps,
        #                                            args.learning_rate_decay_factor, staircase=True,name='exponential_decay_learning_rate')
        lr = tf.placeholder(tf.float32, [])
        with tf.name_scope("create_optimize"):
            # optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)
            # var_list = [v for v in tf.trainable_variables()]
            # print("--------------------------------")
            # print(var_list)
            # print("--------------------------------")
            optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

        saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=10)

        tf.summary.scalar('learning_rate', lr)
        tf.summary.scalar('emd_loss', emd_loss_out)
        # Build the summary Tensor based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

    with tf.Session(graph=graph) as sess:

        sess.run(tf.global_variables_initializer())
        vgg.load_weights(args.pretrain_weights, sess)

        # create queue coordinator
        coord = tf.train.Coordinator()
        thread = tf.train.start_queue_runners(coord=coord, sess=sess)

        # Instantiate a SummaryWriter to output summaries and the Graph.
        summary_writer = tf.summary.FileWriter(
            os.path.join(args.logs_dir, 'train/{}-{}'.format(args.exp_name, timestamp)), sess.graph,
            filename_suffix=args.exp_name)
        summary_test = tf.summary.FileWriter(os.path.join(args.logs_dir, 'test/{}-{}'.format(args.exp_name, timestamp)),
                                             filename_suffix=args.exp_name)
        # global_var = tf.global_variables()
        # var_list = sess.run(global_var)

        import time
        start_time = time.time()
        best_epoch = 0
        base_lr = args.learning_rate
        iters_per_epoch = len(reader.image_list) / args.batch_size
        for step in range(args.iter_max):

            if (step + 1) % (0.5 * args.iter_max) == 0:
                base_lr = base_lr / 5
            if (step + 1) % (0.8 * args.iter_max) == 0:
                base_lr = base_lr / 5

            # base_lr=(base_lr-base_lr*0.001)/args.iter_max*(args)

            image_batch_, label_batch_, mean_std_batch_ = sess.run([image_batch, label_batch, mean_std_batch])
            mean_std_batch_ = mean_std_batch_[:, 0].reshape(-1, 1)
            means_out, mean_hat_out, emd_loss_, l2_loss_, total_loss_, _ = sess.run(
                [means, mean_hat, emd_loss_out, l2_loss, loss, optimizer],
                feed_dict={imgs: image_batch_, scores: label_batch_, means: mean_std_batch_, lr: base_lr})

            if step % iters_per_epoch == 0:
                logger.info(
                    "step %d/%d, the emd loss is %f,l2_loss is %f,total loss is %f, time %f,learning rate: %lf" % (
                    step, args.iter_max, emd_loss_, l2_loss_, total_loss_, (time.time() - start_time), base_lr))
                summary_str = sess.run(summary_op,
                                       feed_dict={imgs: image_batch_, scores: label_batch_, means: mean_std_batch_,
                                                  lr: base_lr})
                # print(means_out.reshape(-1,))
                # print(mean_hat_out)
                # srocc, krocc, plcc, rmse, mse = evaluate_metric(means_out.reshape(-1, ), mean_hat_out)
                # logger.info(
                #     "evaluate train batch SROCC_v: %.3f\t KROCC: %.3f\t PLCC_v: %.3f\t RMSE_v: %.3f\t mse: %.3f\n" % (
                #     srocc, krocc, plcc, rmse, mse))

                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

            if args.save_ckpt_file and step % args.eval_step == 0:  # every epoch save ckpt times
                # saver.save(sess, args.checkpoint_dir + 'iteration_' + str(step) + '.ckpt',write_meta_graph=False)
                save(saver, sess, args.ckpt_dir, step)

                if args.is_eval:
                    score_set = []
                    label_set = []
                    loss_set = []
                    for i in range(len(test_reader.image_list)):
                        image_, score_, mean_std_ = sess.run([test_image, test_score, test_mean_std])
                        # label_set.append(mean_std_[0])
                        label_set.append(mean_std_[:, 0])
                        # mean_std_ = mean_std_[:, 0].reshape(-1, 1)
                        emd_loss_out_, scores_hat_test = sess.run([emd_loss_out, scores_hat],
                                                                  feed_dict={imgs: image_, scores: score_})
                        loss_set.append(emd_loss_out_)
                        # mean,std=scores_stats(scores_hat_)
                        mean_test = mean_score(scores_hat_test)
                        score_set.append(mean_test)
                        if i == 10:
                            summary_str = sess.run(summary_op, feed_dict={imgs: image_, scores: score_, lr: base_lr})
                            summary_test.add_summary(summary_str, step)
                            summary_test.flush()

                    srocc, krocc, plcc, rmse, mse = evaluate_metric(label_set, score_set)
                    print(len(label_set), len(score_set))
                    logger.info(
                        "==============evaluating test datasets :SROCC_v: %.3f\t KROCC: %.3f\t PLCC_v: %.3f\t RMSE_v: %.3f\t mse: %.3f, emd loss is: %f\n" % (
                            srocc, krocc, plcc, rmse, mse, np.mean(loss_set)))

        logger.info("Optimization finish!")
        coord.request_stop()
        coord.join(thread)


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
    logger = setup_logger("TF_NIMA_training_"+args.dataset, output_dir, "train_")
    logger.info(args)

    train(args)


if __name__ == "__main__":
    main()
