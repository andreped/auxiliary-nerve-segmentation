import tensorflow as tf
from keras.objectives import mean_squared_error
import keras.backend as K
from keras.models import load_model

from autoencoder import create_autoencoder


def dice_loss(target, output, epsilon=1e-10):
    """Sørensen–Dice coefficient for comparing the similarity of two distributions,
    usually be used for binary image segmentation i.e. labels are binary.
    The coefficient = [0, 1], 1 if totally match.
    Taken from: http://tensorlayer.readthedocs.io/en/latest/_modules/tensorlayer/cost.html#dice_coe

    Parameters
    -----------
    output : tensor
        A distribution with shape: [batch_size, ....], (any dimensions).
    target : tensor
        A distribution with shape: [batch_size, ....], (any dimensions).
    epsilon : float
        An optional name to attach to this layer.

    Examples
    ---------
    >>> outputs = tl.act.pixel_wise_softmax(network.outputs)
    >>> dice_loss = 1 - tl.cost.dice_coe(outputs, y_, epsilon=1e-5)

    References
    -----------
    - `wiki-dice <https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient>`_
    """
    #output = output[:, :, :, 1:]  # Drop background class
    #target = target[:, :, :, 1:]  # Drop background class
    output1 = output[:, :, :, 1]
    target1 = target[:, :, :, 1]
    intersection1 = tf.reduce_sum(output1 * target1)
    union1 = tf.reduce_sum(output1 * output1) + tf.reduce_sum(target1 * target1)
    dice1 = 2 * intersection1 / union1
    output2 = output[:, :, :, 2]
    target2 = target[:, :, :, 2]
    intersection2 = tf.reduce_sum(output2 * target2)
    union2 = tf.reduce_sum(output2 * output2) + tf.reduce_sum(target2 * target2)
    dice2 = 2 * intersection2 / union2
    output3 = output[:, :, :, 3]
    target3 = target[:, :, :, 3]
    intersection3 = tf.reduce_sum(output3 * target3)
    union3 = tf.reduce_sum(output3 * output3) + tf.reduce_sum(target3 * target3)
    dice3 = 2 * intersection3 / union3
    output4 = output[:, :, :, 4]
    target4 = target[:, :, :, 4]
    intersection4 = tf.reduce_sum(output4 * target4)
    union4 = tf.reduce_sum(output4 * output4) + tf.reduce_sum(target4 * target4)
    dice4 = 2 * intersection4 / union4
    output5 = output[:, :, :, 5]
    target5 = target[:, :, :, 5]
    intersection5 = tf.reduce_sum(output5 * target5)
    union5 = tf.reduce_sum(output5 * output5) + tf.reduce_sum(target5 * target5)
    dice5 = 2 * intersection5 / union5
    dice = (dice1 + dice2 + dice3 + dice4 + dice5) / 5.0
    return tf.clip_by_value(1 - dice, 0, 1.0 - epsilon)


def custom_fcn_loss(y_true, y_pred):
    # This simply applies weighting
    # Adopted from https://github.com/guojiyao/deep_fcn/blob/master/loss.py
    labels = y_true
    logits = y_pred

    logits = tf.reshape(logits, (-1, 6))
    labels = tf.to_float(tf.reshape(labels, (-1, 6)))
    # With class weights:
    class_weights = [1, 10, 10, 10, 10, 10]
    cross_entropy = -tf.reduce_sum(tf.multiply(labels * tf.log(logits), class_weights), axis=1)
    # Without class wights:
    #cross_entropy = -tf.reduce_sum(labels * tf.log(logits), axis=1)
    return tf.reduce_mean(cross_entropy)

def get_encoder(input_tensor):
    # Have to disable learning phase while creating encoder or we get issues form BatchNormalization
    K.set_learning_phase(0)
    model = create_autoencoder((256, 256, 6), input_tensor=input_tensor, encoder_only=True)
    model.load_weights('models/shape_model_weights.hd5', by_name=True)

    # Freeze weights of encoder
    model.trainable = False
    for layer in model.layers:
        layer.trainable = False

    K.set_learning_phase(1)
    return model.outputs[0]


def dice_loss_with_shape_model_regularization(y_true, y_pred):

    alpha = 0.01  # Weighting of shape model

    # Get encoder part of shape model
    return dice_loss(y_true, y_pred) + \
           alpha * mean_squared_error(get_encoder(y_true),
                                      get_encoder(y_pred))