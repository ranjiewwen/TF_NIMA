#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
created checkpoint.py by rjw at 19-1-8 in WHU.
"""

import os

def save(saver, sess, logdir, step):
	'''Save weights.

	Args:
	  saver: TensorFlow Saver object.
	  sess: TensorFlow session.
	  logdir: path to the snapshots directory.
	  step: current training step.
	'''
	model_name = 'model.ckpt'
	checkpoint_path = os.path.join(logdir, model_name)

	if not os.path.exists(logdir):
		os.makedirs(logdir)
	saver.save(sess, checkpoint_path, global_step=step,write_meta_graph=False)
	print('The checkpoint has been created.')


def load(saver, sess, ckpt_path):
	'''Load trained weights.

	Args:
	  saver: TensorFlow Saver object.
	  sess: TensorFlow session.
	  ckpt_path: path to checkpoint file with parameters.
	'''
	saver.restore(sess, ckpt_path)


	print("Restored model parameters from {}".format(ckpt_path))


import tensorflow as tf
import logging

def print_num_of_total_parameters(output_detail=False, output_to_logging=False):
	total_parameters = 0
	parameters_string = ""

	for variable in tf.trainable_variables():

		shape = variable.get_shape()
		variable_parameters = 1
		for dim in shape:
			variable_parameters *= dim.value
		total_parameters += variable_parameters
		if len(shape) == 1:
			parameters_string += ("%s %d, " % (variable.name, variable_parameters))
		else:
			parameters_string += ("%s %s=%d, " % (variable.name, str(shape), variable_parameters))

	if output_to_logging:
		if output_detail:
			logging.info(parameters_string)
		logging.info("Total %d variables, %s params" % (len(tf.trainable_variables()), "{:,}".format(total_parameters)))
	else:
		if output_detail:
			print(parameters_string)
		print("Total %d variables, %s params" % (len(tf.trainable_variables()), "{:,}".format(total_parameters)))
