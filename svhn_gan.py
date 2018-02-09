import tensorflow as tf
import nn  # OpenAI implemetation of weightnormalization (Salimans & Kingma)

init_kernel = tf.random_normal_initializer(mean=0, stddev=0.05)


def leakyReLu(x, alpha=0.2, name=None):
    if name:
        with tf.variable_scope(name):
            return _leakyReLu_impl(x, alpha)
    else:
        return _leakyReLu_impl(x, alpha)


def _leakyReLu_impl(x, alpha):
    return tf.nn.relu(x) - (alpha * tf.nn.relu(-x))

def gaussian_noise_layer(input_layer, std):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
    return input_layer + noise


def discriminator(inp, is_training, init=False, reuse=False, getter =None):
    with tf.variable_scope('discriminator_model', reuse=reuse,custom_getter=getter):
        counter = {}
        x = tf.reshape(inp, [-1, 32, 32, 3])

        x = tf.layers.dropout(x, rate=0.2, training=is_training, name='dropout_0')

        x = nn.conv2d(x, 64, nonlinearity=leakyReLu, init=init, counters=counter)
        x = nn.conv2d(x, 64, nonlinearity=leakyReLu, init=init, counters=counter)
        x = nn.conv2d(x, 64, stride=[2, 2], nonlinearity=leakyReLu, init=init, counters=counter)

        x = tf.layers.dropout(x, rate=0.5, training=is_training, name='dropout_1')

        x = nn.conv2d(x, 128, nonlinearity=leakyReLu, init=init, counters=counter)
        x = nn.conv2d(x, 128, nonlinearity=leakyReLu, init=init, counters=counter)
        x = nn.conv2d(x, 128, stride=[2, 2], nonlinearity=leakyReLu, init=init, counters=counter)

        x = tf.layers.dropout(x, rate=0.5, training=is_training, name='dropout_2')

        x = nn.conv2d(x, 128, pad='VALID', nonlinearity=leakyReLu, init=init, counters=counter)
        x = nn.nin(x, 128, counters=counter, nonlinearity=leakyReLu, init=init)
        x = nn.nin(x, 128, counters=counter, nonlinearity=leakyReLu, init=init)
        x = tf.layers.max_pooling2d(x, pool_size=6, strides=1,
                                    name='avg_pool_0')
        x = tf.squeeze(x, [1, 2])

        intermediate_layer = x

        logits = nn.dense(x, 10, nonlinearity=None, init=init, counters=counter, init_scale=0.1)

        return logits, intermediate_layer


def generator(z_seed, is_training, init=False,reuse=False):
    with tf.variable_scope('generator_model', reuse=reuse):
        counter = {}
        x = z_seed
        with tf.variable_scope('dense_1'):
            x = tf.layers.dense(x, units=4 * 4 * 512, kernel_initializer=init_kernel)
            x = tf.layers.batch_normalization(x, training=is_training, name='batchnorm_1')
            x = tf.nn.relu(x)

        x = tf.reshape(x, [-1, 4, 4, 512])

        with tf.variable_scope('deconv_1'):
            x = tf.layers.conv2d_transpose(x, 256, [5, 5], strides=[2, 2], padding='SAME', kernel_initializer=init_kernel)
            x = tf.layers.batch_normalization(x, training=is_training, name='batchnorm_2')
            x = tf.nn.relu(x)

        with tf.variable_scope('deconv_2'):
            x = tf.layers.conv2d_transpose(x, 128, [5, 5], strides=[2, 2], padding='SAME', kernel_initializer=init_kernel)
            x = tf.layers.batch_normalization(x, training=is_training, name='batchnormn_3')
            x = tf.nn.relu(x)

        with tf.variable_scope('deconv_3'):
            output = nn.deconv2d(x, num_filters=3, filter_size=[5, 5], stride=[2, 2], nonlinearity=tf.tanh, init=init,
                                 counters=counter, init_scale=0.1)
        return output
