#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
created VGG_model_K.py by rjw at 19-1-10 in WHU.
"""

import keras
from keras.models import Model
from keras.layers import Dense, Dropout
#from keras.applications.mobilenet import MobileNet
from keras.applications.vgg16 import VGG16


def vgg_model():
    base_model = VGG16(include_top=True, pooling='avg')
    base_model.layers.pop()
    base_model_output = base_model.layers[-1].output
    base_model.layers[-1].outbound_nodes = []
    # for layer in base_model.layers:
    #     #print(layer)
    #     layer.trainable = False

    x = Dropout(0.75)(base_model_output)
    x = Dense(10, activation='softmax')(x)

    model = Model(base_model.input, x)
    model.summary()
    return model

