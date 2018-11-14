import tensorflow as tf
import numpy as np
import time

from tensorflow.contrib.rnn import GRUCell
from util.infolog import log


def prenet(inputs, is_training, layer_sizes, scope=None):
    x = inputs
    drop_rate = 0.5 if is_training else 0.0
    with tf.variable_scope(scope or 'prenet'):
        for i, size in enumerate(layer_sizes):
            dense = tf.layers.dense(x, units=size, activation=tf.nn.relu, name='dense_%d' % (i + 1))
            x = tf.layers.dropout(dense, rate=drop_rate, training=is_training, name='dropout_%d' % (i + 1))
    return x


def encoder_cbhg(inputs, input_lengths, is_training, depth):
    input_channels = inputs.get_shape()[2]
    return cbhg(
        inputs,
        input_lengths,
        is_training,
        scope='encoder_cbhg',
        K=16,
        projections=[128, input_channels],
        depth=depth)


def post_cbhg(inputs, input_dim, is_training, depth):
    return cbhg(
        inputs,
        None,
        is_training,
        scope='post_cbhg',
        K=8,
        projections=[256, input_dim],
        depth=depth)


def cbhg(inputs, input_lengths, is_training, scope, K, projections, depth):
    with tf.variable_scope(scope):
        with tf.variable_scope('conv_bank'):
            # Convolution bank: concatenate on the last axis to stack channels from all convolutions
            conv_outputs = tf.concat(
                [conv1d(inputs, k, 128, tf.nn.relu, is_training, 'conv1d_%d' % k) for k in range(1, K + 1)],
                axis=-1
            )

        # Maxpooling:
        maxpool_output = tf.layers.max_pooling1d(
            conv_outputs,
            pool_size=2,
            strides=1,
            padding='same')

        # Two projection layers:
        proj1_output = conv1d(maxpool_output, 3, projections[0], tf.nn.relu, is_training, 'proj_1')
        proj2_output = conv1d(proj1_output, 3, projections[1], None, is_training, 'proj_2')

        # Residual connection:
        highway_input = proj2_output + inputs

        half_depth = depth // 2
        assert half_depth * 2 == depth, 'encoder and postnet depths must be even.'

        # Handle dimensionality mismatch:
        if highway_input.shape[2] != half_depth:
            highway_input = tf.layers.dense(highway_input, half_depth)

        # 4-layer HighwayNet:
        for i in range(4):
            highway_input = highwaynet(highway_input, 'highway_%d' % (i + 1), half_depth)
        rnn_input = highway_input

        # Bidirectional RNN
        outputs, states = tf.nn.bidirectional_dynamic_rnn(
            GRUCell(half_depth),
            GRUCell(half_depth),
            rnn_input,
            sequence_length=input_lengths,
            dtype=tf.float32)
        return tf.concat(outputs, axis=2)  # Concat forward and backward


def highwaynet(inputs, scope, depth):
    with tf.variable_scope(scope):
        H = tf.layers.dense(
            inputs,
            units=depth,
            activation=tf.nn.relu,
            name='H')
        T = tf.layers.dense(
            inputs,
            units=depth,
            activation=tf.nn.sigmoid,
            name='T',
            bias_initializer=tf.constant_initializer(-1.0))
        return H * T + inputs * (1.0 - T)


def conv1d(inputs, kernel_size, channels, activation, is_training, scope):
    with tf.variable_scope(scope):
        conv1d_output = tf.layers.conv1d(
            inputs,
            filters=channels,
            kernel_size=kernel_size,
            activation=activation,
            padding='same')
        return tf.layers.batch_normalization(conv1d_output, training=is_training)


VGG_MEAN = [103.939, 116.779, 123.68]


# noinspection PyMethodMayBeStatic
class Vgg19:
    def __init__(self, vgg19_npy_path):
        self.data_dict = np.load(vgg19_npy_path, encoding='latin1').item()

    def build(self, rgb):
        """
        load variable from npy to build the VGG
        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """

        start_time = time.time()
        log('Building VGG19. Started at: %ds' % start_time)
        rgb_scaled = tf.cast(rgb, tf.float32) * 255.0

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        self.conv1_1 = self.conv_layer(bgr, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.conv3_4 = self.conv_layer(self.conv3_3, "conv3_4")
        self.pool3 = self.max_pool(self.conv3_4, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.conv4_4 = self.conv_layer(self.conv4_3, "conv4_4")
        self.pool4 = self.max_pool(self.conv4_4, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        self.conv5_4 = self.conv_layer(self.conv5_3, "conv5_4")
        self.pool5 = self.max_pool(self.conv5_4, 'pool5')

        self.fc6 = self.fc_layer(self.pool5, "fc6")
        assert self.fc6.get_shape().as_list()[1:] == [4096]
        self.relu6 = tf.nn.relu(self.fc6)

        self.fc7 = self.fc_layer(self.relu6, "fc7")
        self.relu7 = tf.nn.relu(self.fc7)

        self.fc8 = self.fc_layer(self.relu7, "fc8")

        log("finished building VGG19 in %ds" % (time.time() - start_time))

        return self.fc8

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def fc_layer(self, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")

    def get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name="weights")