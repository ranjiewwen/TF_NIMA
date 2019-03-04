#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
created graph_test.py by rjw at 19-1-9 in WHU.
"""

import tensorflow as tf
import numpy as np

c = tf.constant(value=1)
print(c.graph)
print(tf.get_default_graph)