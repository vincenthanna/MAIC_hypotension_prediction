# Multiple ResNet + SENet model by kernel sizes
#
# forward path:
#   Multiple ResNets -> concat with democratic data -> ResNet again -> Dense
# 
# ==============================================================================
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.models import Model, load_model
from keras.layers import Dense, Conv1D, MaxPooling1D, GlobalMaxPool1D, BatchNormalization, Dropout, Activation, Add, Layer, GlobalAveragePooling1D, Input, Concatenate,Reshape, Dense, multiply, add, Permute, Lambda
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras import optimizers
from keras.utils.vis_utils import plot_model
import pandas as pd
import numpy as np

def squeeze_excite_block(input_tensor, ratio=16):
    """ Create a channel-wise squeeze-excite block
    Args:
        input_tensor: input Keras tensor
        ratio: number of output filters
    Returns: a Keras tensor
    References
    -   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
    """
    init = input_tensor
    channel_axis = 1 if tf.keras.backend.image_data_format() == "channels_first" else -1
    filters = init.shape[channel_axis]
    se_shape = (1, filters)

    se = keras.layers.GlobalAveragePooling1D()(init)
    se = Reshape(se_shape)(se)    
    se = keras.layers.Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = keras.layers.Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)    

    x = multiply([init, se])
    return x


def resnet_basic_block(inputs, filters, kernel_size, strides):
    x = inputs
    shortcut = inputs
    print("x.shape=", x.shape)
    x = keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    
    x = keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=1, padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)    

    x = squeeze_excite_block(x)
    
    shortcut = keras.layers.Conv1D(filters=filters, kernel_size=1, strides=strides, padding='same', activation='relu')(shortcut)
    shortcut = keras.layers.BatchNormalization()(shortcut)

    x = keras.layers.Add()([shortcut, x])
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    return x


def resnet(inputs, kernel_size, filter_sizes):
    x = inputs
    for fsize in filter_sizes:
        x = resnet_basic_block(inputs=x, filters=fsize, kernel_size=kernel_size, strides=2)
    return x


def resnet_demo_net(inputs, kernel_sizes, filter_sizes):
    
    # you have to use keras layer class for splitting.
    # just splitting like array doesn't work    
    
    # bp data
    input_bp = Lambda(lambda x: x[:, :2000])(inputs)
    input_bp = tf.keras.backend.expand_dims(input_bp, axis=-1)
    
    # demographic
    input_demo = Lambda(lambda x: x[:, 2000:])(inputs)

    sub_models = []
    for kernel_size in kernel_sizes:        
        sm = resnet(input_bp, kernel_size=kernel_size, filter_sizes=filter_sizes)
        sm = keras.layers.GlobalAveragePooling1D()(sm)        
        sub_models.append(sm)

    if len(sub_models) > 1:        
        x = keras.layers.Concatenate()(sub_models)
    else:
        x = sub_models[0]
    
    x = Concatenate()([x, input_demo])

    x = tf.expand_dims(x, -1)

    x = resnet_basic_block(x, filters=16, kernel_size=3, strides=2)
    x = resnet_basic_block(x, filters=32, kernel_size=3, strides=2)
    x = resnet_basic_block(x, filters=64, kernel_size=3, strides=2)
    
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(units=1, activation='sigmoid')(x)

    return x