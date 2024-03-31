import tensorflow as tf

def dice_metric(object_index, threshold=0.1, epsilon=1e-10):
    """Non-differentiable Sørensen–Dice coefficient for comparing the similarity of two distributions,
    usually be used for binary image segmentation i.e. labels are binary.
    The coefficient = [0, 1], 1 if totally match.

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
    >>> outputs = pixel_wise_softmax(network.outputs)
    >>> dice_loss = 1 - dice_coe(outputs, y_, epsilon=1e-5)

    References
    -----------
    - `wiki-dice <https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient>`_
    """
    def dice_coe(output, target):
        output = output[:, :, :, object_index]
        target = target[:, :, :, object_index]
        output = tf.cast(output > threshold, dtype=tf.float32)
        target = tf.cast(target > threshold, dtype=tf.float32)
        inse = tf.reduce_sum(output * target)
        l = tf.reduce_sum(output * output)
        r = tf.reduce_sum(target * target)
        dice = 2 * inse / (l + r)
        if epsilon == 0:
            return dice
        else:
            return tf.clip_by_value(dice, 0, 1.0-epsilon)

    return dice_coe