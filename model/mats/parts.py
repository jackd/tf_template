import tensorflow as tf


def encode_voxels(voxels, mode):
    training = mode == tf.estimator.ModeKeys.TRAIN
    with tf.variable_scope('encoder'):
        x = voxels
        for i in range(4):
            x = tf.layers.conv3d(x, 3, 1, activation=tf.nn.relu)
            x = tf.layers.batch_normalization(
                x, scale=False, training=training)
            x = tf.layers.max_pooling3d(x, 3, 2)
    return x


def decode(latent):
    with tf.variable_scope('deocder'):
        raise NotImplementedError('TODO')


if __name__ == '__main__':
    voxels = tf.placeholder(shape=(2, 32, 32, 32, 1), dtype=tf.float32)
    encoding = encode_voxels(voxels, 'train')
    print(tf.get_collection(tf.GraphKeys.VARIABLES, scope='encoder'))
