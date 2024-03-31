from tensorflow.python.keras.layers import Input, Dense, Convolution2D, MaxPooling2D, Dropout, Flatten, SpatialDropout2D, \
    ZeroPadding2D, Activation, AveragePooling2D, UpSampling2D, BatchNormalization, ConvLSTM2D, \
    TimeDistributed, Concatenate, Lambda, Reshape, UpSampling3D, Convolution3D, MaxPooling3D, SpatialDropout3D
from tensorflow.python.keras.models import Model
import tensorflow as tf


def convolution_block_2d(x, nr_of_convolutions, use_bn=False, spatial_dropout=None):
    for i in range(2):
        x = Convolution2D(nr_of_convolutions, 3, padding='same')(x)
        if use_bn:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)
        if spatial_dropout:
            x = SpatialDropout2D(spatial_dropout)(x)

    return x


def encoder_block_2d(x, nr_of_convolutions, use_bn=False, spatial_dropout=None):

    x_before_downsampling = convolution_block_2d(x, nr_of_convolutions, use_bn, spatial_dropout)
    x = MaxPooling2D((2, 2))(x_before_downsampling)

    return x, x_before_downsampling


def decoder_block_2d(x, nr_of_convolutions, cross_over_connection=None, use_bn=False, spatial_dropout=None):

    x = UpSampling2D((2, 2))(x)
    if cross_over_connection is not None:
        x = Concatenate()([cross_over_connection, x])
    x = convolution_block_2d(x, nr_of_convolutions, use_bn, spatial_dropout)

    return x

def convolution_block_3d(x, nr_of_convolutions, use_bn=False, spatial_dropout=None):
    for i in range(2):
        x = Convolution3D(nr_of_convolutions, 3, padding='same')(x)
        if use_bn:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)
        if spatial_dropout:
            x = SpatialDropout3D(spatial_dropout)(x)

    return x


def encoder_block_3d(x, nr_of_convolutions, use_bn=False, spatial_dropout=None):

    x_before_downsampling = convolution_block_3d(x, nr_of_convolutions, use_bn, spatial_dropout)
    downsample = [2, 2, 2]
    for i in range(1, 4):
        if x.shape[i] <= 4:
            downsample[i-1] = 1

    x = MaxPooling3D(downsample)(x_before_downsampling)

    return x, x_before_downsampling


def decoder_block_3d(x, nr_of_convolutions, cross_over_connection=None, use_bn=False, spatial_dropout=None):

    upsample = [2, 2, 2]
    if cross_over_connection is not None:
        for i in range(1, 4):
            if cross_over_connection.shape[i] == x.shape[i]:
                upsample[i-1] = 1
    x = UpSampling3D(upsample)(x)
    if cross_over_connection is not None:
        x = Concatenate()([cross_over_connection, x])
    x = convolution_block_3d(x, nr_of_convolutions, use_bn, spatial_dropout)

    return x


def encoder_block(x, nr_of_convolutions, use_bn=False, spatial_dropout=None, dims=2):
    if dims == 2:
        return encoder_block_2d(x, nr_of_convolutions, use_bn, spatial_dropout)
    elif dims == 3:
        return encoder_block_3d(x, nr_of_convolutions, use_bn, spatial_dropout)
    else:
        raise ValueError


def decoder_block(x, nr_of_convolutions, cross_over_connection=None, use_bn=False, spatial_dropout=None, dims=2):
    if dims == 2:
        return decoder_block_2d(x, nr_of_convolutions, cross_over_connection, use_bn, spatial_dropout)
    elif dims == 3:
        return decoder_block_3d(x, nr_of_convolutions, cross_over_connection, use_bn, spatial_dropout)
    else:
        raise ValueError


def convolution_block(x, nr_of_convolutions, use_bn=False, spatial_dropout=None, dims=2):
    if dims == 2:
        return convolution_block_2d(x, nr_of_convolutions, use_bn, spatial_dropout)
    elif dims == 3:
        return convolution_block_3d(x, nr_of_convolutions, use_bn, spatial_dropout)
    else:
        raise ValueError

class VGGnet():
    def __init__(self, input_shape, nb_classes):
        if len(input_shape) != 3:
            raise ValueError('Input shape must have 3 dimensions')
        if nb_classes <= 1:
            raise ValueError('Classes must be > 1')
        self.nb_classes = nb_classes
        self.input_shape = input_shape
        self.convolutions = None
        self.use_bn = True
        self.spatial_dropout = None
        self.dense_dropout = 0.5
        self.dense_size = 1024

    def set_dense_size(self, size):
        self.dense_size = size

    def set_dense_dropout(self, dropout):
        self.dense_dropout = dropout

    def set_spatial_dropout(self, dropout):
        self.spatial_dropout = dropout

    def set_convolutions(self, convolutions):
        if len(convolutions) != self.get_depth():
            raise ValueError('Nr of convolutions must have length ' + str(self.get_depth()*2 + 1))
        self.convolutions = convolutions

    def get_depth(self):
        init_size = min(self.input_shape[0], self.input_shape[1])
        size = init_size
        depth = 0
        while size % 2 == 0 and size > 4:
            size /= 2
            depth += 1

        return depth + 1

    def create(self):
        """
        Create model and return it

        :return: keras model
        """

        input_layer = Input(shape=self.input_shape)
        x = input_layer

        init_size = min(self.input_shape[:-1])
        size = init_size

        convolutions = self.convolutions
        if convolutions is None:
            # Create convolutions
            convolutions = []
            nr_of_convolutions = 8
            for i in range(self.get_depth()):
                convolutions.append(nr_of_convolutions)
                nr_of_convolutions *= 2

        i = 0
        while size % 2 == 0 and size > 4:
            x, _ = encoder_block_2d(x, convolutions[i], self.use_bn, self.spatial_dropout)
            size /= 2
            i += 1

        x = convolution_block_2d(x, convolutions[i], self.use_bn, self.spatial_dropout)

        x = Flatten(name="flatten")(x)
        x = Dense(self.dense_size, activation='relu')(x)
        x = Dropout(self.dense_dropout)(x)
        x = Dense(self.dense_size, activation='relu')(x)
        x = Dropout(self.dense_dropout)(x)
        x = Dense(self.nb_classes, activation='softmax')(x)

        return Model(inputs=input_layer, outputs=x)


class Unet():
    def __init__(self, input_shape, nb_classes):
        if len(input_shape) != 3 and len(input_shape) != 4:
            raise ValueError('Input shape must have 3 or 4 dimensions')
        if len(input_shape) == 3:
            self.dims = 2
        else:
            self.dims = 3
        if nb_classes <= 1:
            raise ValueError('Segmentation classes must be > 1')
        self.input_shape = input_shape
        self.nb_classes = nb_classes
        self.convolutions = None
        self.encoder_use_bn = True
        self.decoder_use_bn = True
        self.encoder_spatial_dropout = None
        self.decoder_spatial_dropout = None

    def set_convolutions(self, convolutions):
        if len(convolutions) != self.get_depth()*2 + 1:
            raise ValueError('Nr of convolutions must have length ' + str(self.get_depth()*2 + 1))
        self.convolutions = convolutions

    def get_depth(self):
        init_size = max(self.input_shape[:-1])
        size = init_size
        depth = 0
        while size % 2 == 0 and size > 4:
            size /= 2
            depth += 1

        return depth

    def get_dice_loss(self):
        def dice_loss(target, output, epsilon=1e-10):
            smooth = 1.
            dice = 0

            for object in range(1, self.nb_classes):
                if self.dims == 2:
                    output1 = output[:, :, :, object]
                    target1 = target[:, :, :, object]
                else:
                    output1 = output[:, :, :, :, object]
                    target1 = target[:, :, :, :, object]
                intersection1 = tf.reduce_sum(output1 * target1)
                union1 = tf.reduce_sum(output1 * output1) + tf.reduce_sum(target1 * target1)
                dice += (2. * intersection1 + smooth) / (union1 + smooth)

            dice /= (self.nb_classes - 1)

            return tf.clip_by_value(1. - dice, 0., 1. - epsilon)

        return dice_loss


    def create(self):
        """
        Create model and return it

        :return: keras model
        """

        input_layer = Input(shape=self.input_shape)
        x = input_layer

        init_size = max(self.input_shape[:-1])
        size = init_size

        convolutions = self.convolutions
        if convolutions is None:
            # Create convolutions
            convolutions = []
            nr_of_convolutions = 8
            for i in range(self.get_depth()):
                convolutions.append(int(nr_of_convolutions))
                nr_of_convolutions *= 2
            convolutions.append(int(nr_of_convolutions))
            for i in range(self.get_depth()):
                convolutions.append(int(nr_of_convolutions))
                nr_of_convolutions /= 2

        connection = {}
        i = 0
        while size % 2 == 0 and size > 4:
            x, connection[size] = encoder_block(x, convolutions[i], self.encoder_use_bn, self.encoder_spatial_dropout, self.dims)
            size /= 2
            i += 1

        x = convolution_block(x, convolutions[i], self.encoder_use_bn, self.encoder_spatial_dropout, self.dims)
        i += 1

        while size < init_size:
            size *= 2
            x = decoder_block(x, convolutions[i], connection[size], self.decoder_use_bn, self.decoder_spatial_dropout, self.dims)
            i += 1

        if self.dims == 2:
            x = Convolution2D(self.nb_classes, 1, activation='softmax')(x)
        else:
            x = Convolution3D(self.nb_classes, 1, activation='softmax')(x)

        return Model(inputs=input_layer, outputs=x)
