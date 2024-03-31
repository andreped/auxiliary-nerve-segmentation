from keras.layers import Input, Conv2D, Conv2DTranspose, Flatten, Dense, Reshape, BatchNormalization, Activation
from keras.models import Model
from math import sqrt
import tensorflow as tf


def create_autoencoder(shape, code_length=64, input_tensor=None, encoder_only=False):
    if encoder_only:
        print(input_tensor)
        input_layer = Input(shape=shape, tensor=input_tensor)
    else:
        input_layer = Input(shape=shape)
    x = Conv2D(16, (3, 3), activation='relu', strides=(2, 2), name='sm_conv_1')(input_layer)
    x = BatchNormalization(name='sm_bn_1')(x)
    x = Conv2D(16, (3, 3), activation='relu', name='sm_conv_2')(x)
    x = BatchNormalization(name='sm_bn_2')(x)
    x = Conv2D(32, (3, 3), activation='relu', strides=(2, 2), name='sm_conv_3')(x)
    x = BatchNormalization(name='sm_bn_3')(x)
    x = Conv2D(32, (3, 3), activation='relu', name='sm_conv_4')(x)
    x = BatchNormalization(name='sm_bn_4')(x)
    x = Conv2D(64, (3, 3), activation='relu', strides=(2, 2), name='sm_conv_5')(x)
    x = BatchNormalization(name='sm_bn_5')(x)
    x = Conv2D(64, (3, 3), activation='relu', name='sm_conv_6')(x)
    x = BatchNormalization(name='sm_bn_6')(x)
    x = Conv2D(128, (3, 3), activation='relu', strides=(2, 2), name='sm_conv_7')(x)
    x = BatchNormalization(name='sm_bn_7')(x)
    x = Conv2D(128, (3, 3), activation='relu', name='sm_conv_8')(x)
    x = BatchNormalization(name='sm_bn_8')(x)
    x = Conv2D(1, (3, 3), activation='relu', strides=(2, 2), name='sm_conv_9')(x)
    last_conv = x
    x = BatchNormalization(name='sm_bn_9')(x)

    x = Flatten()(x)
    x = Dense(code_length, name='low_dimensional_shape_model')(x)

    if not encoder_only:
        last_conv_width = int(last_conv.shape[1])
        last_conv_height = int(last_conv.shape[2])
        x = Dense(last_conv_width*last_conv_height*64, activation='relu')(x)
        x = Reshape((last_conv_width, last_conv_height, 64))(x)

        x = Conv2DTranspose(128, (10, 10),  activation='relu', strides=(2, 2))(x)
        x = Conv2D(128, (3, 3), activation='relu')(x)
        #x = BatchNormalization()(x)
        x = Conv2DTranspose(64, (4, 4),  activation='relu', strides=(2, 2))(x)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        #x = BatchNormalization()(x)
        x = Conv2DTranspose(32, (4, 4),  activation='relu', strides=(2, 2))(x)
        x = Conv2D(32, (3, 3), activation='relu')(x)
        #x = BatchNormalization()(x)
        x = Conv2DTranspose(16, (4, 4),  activation='relu', strides=(2, 2))(x)
        x = Conv2D(16, (3, 3), activation='relu')(x)
        #x = BatchNormalization()(x)
        x = Conv2DTranspose(16, (4, 4),  activation='relu', strides=(2, 2))(x)
        x = Conv2D(6, (3, 3), activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=x)

    return model