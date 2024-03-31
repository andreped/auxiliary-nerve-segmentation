from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, Dropout, Flatten, Merge, SpatialDropout2D, \
    Deconvolution2D, ZeroPadding2D, Activation, AveragePooling2D, UpSampling2D, BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model
import tensorflow as tf


def create_fc_network(nr_of_objects, input_shape, extra_deep=False):
    inputs = Input(shape=input_shape, name='input_image')
    # 256x256
    x = Convolution2D(8, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SpatialDropout2D(0.25)(x)
    x = Convolution2D(8, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SpatialDropout2D(0.25)(x)
    level256 = x
    x = MaxPooling2D()(x)
    # 128x128
    x = Convolution2D(16, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SpatialDropout2D(0.25)(x)
    x = Convolution2D(16, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SpatialDropout2D(0.25)(x)
    level128 = x
    x = MaxPooling2D()(x)
    # 64x64
    x = Convolution2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SpatialDropout2D(0.25)(x)
    x = Convolution2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SpatialDropout2D(0.25)(x)
    level64 = x
    x = MaxPooling2D()(x)
    # 32x32
    x = Convolution2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SpatialDropout2D(0.25)(x)
    x = Convolution2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SpatialDropout2D(0.25)(x)
    level32 = x
    x = MaxPooling2D()(x)
    # 16 x 16
    x = Convolution2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SpatialDropout2D(0.25)(x)
    x = Convolution2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SpatialDropout2D(0.25)(x)
    level16 = x
    x = MaxPooling2D()(x)
    # 8 x 8
    x = Convolution2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SpatialDropout2D(0.25)(x)
    x = Convolution2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SpatialDropout2D(0.25)(x)

    if extra_deep:
        level8 = x
        x = MaxPooling2D()(x)
        # 4 x 4
        x = Convolution2D(256, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Convolution2D(256, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = UpSampling2D(size=(2, 2))(x)
        x = Concatenate()([level8, x])
        x = Convolution2D(128, (3, 3), activation='relu', padding='same')(x)
        x = Convolution2D(128, (3, 3), activation='relu', padding='same')(x)

    # Upsample to 16x16
    x = UpSampling2D(size=(2, 2))(x)
    x = Concatenate()([level16, x])
    x = Convolution2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Convolution2D(128, (3, 3), activation='relu', padding='same')(x)
    # Upsample to 32x32
    x = UpSampling2D(size=(2, 2))(x)
    x = Concatenate()([level32, x])
    x = Convolution2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Convolution2D(64, (3, 3), activation='relu', padding='same')(x)
    # Upsample to 64x64
    x = UpSampling2D(size=(2, 2))(x)
    x = Concatenate()([level64, x])
    x = Convolution2D(32, (3, 3), activation='relu', padding='same')(x)
    x = Convolution2D(32, (3, 3), activation='relu', padding='same')(x)
    # Upsample to 128x128
    x = UpSampling2D(size=(2, 2))(x)
    x = Concatenate()([level128, x])
    x = Convolution2D(16, (3, 3), activation='relu', padding='same')(x)
    x = Convolution2D(16, (3, 3), activation='relu', padding='same')(x)
    # Upsample to 256x256
    x = UpSampling2D(size=(2, 2))(x)
    x = Concatenate()([level256, x])
    x = Convolution2D(8, (3, 3), activation='relu', padding='same')(x)
    x = Convolution2D(8, (3, 3), activation='relu', padding='same')(x)
    x = Convolution2D(nr_of_objects, (1, 1), activation='softmax')(x)

    model = Model(inputs=inputs, outputs=x)
    #print(model.summary(line_length=150))

    return model
