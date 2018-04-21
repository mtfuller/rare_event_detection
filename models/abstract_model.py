import time
import tensorflow as tf
import numpy as np
from tensorflow.contrib import layers
import math

DEFAULT_INPUT_NAME = 'input'
DEFAULT_OUTPUT_NAME = 'output'

class AbstractModel:
    def __init__(self, input_shape, **kwargs):
        self.graph = tf.Graph()
        self.session = tf.Session(config=tf.ConfigProto(), graph=self.graph)
        self.input_shape = input_shape
        self.input_name = kwargs.get('input_name', DEFAULT_INPUT_NAME)
        self.output_name = kwargs.get('output_name', DEFAULT_OUTPUT_NAME)
        self.prob = tf.placeholder_with_default(1.0, shape=())
        self.build_model()

    def predict(self, _input):
        """Returns an anomaly score based on the given set of video segment.

        Runs a forward propagation on the network, which computes an anomaly score between 0 and 1.

        Args:
            input: A tensor of some dimension that

        Returns:
            A numpy array of float values that has a size equal to the given number of segments. For example, if a set
            of 3 video segments was passed in, the function would return an array of size 3, where each element is an
            anomaly score of the corresponding video segment.
        """
        with self.graph.as_default():
            start = time.time()
            print("INPUT SHAPE: %s" % (str(_input.shape)))
            input_list = np.array_split(_input, math.ceil(_input.shape[0]/16))
            for i in range(len(input_list)):
              input_list[i] = self.session.run(self.net, feed_dict={self.inputs: input_list[i]})
            output = np.vstack(input_list)
            end = time.time()
            return output, (end-start)

    def build_model(self):
        with self.graph.as_default():
            self.initializer = layers.xavier_initializer()
            self.net = self.inputs = tf.placeholder(tf.float32, self.input_shape, name=self.input_name)
            self.prob = tf.placeholder_with_default(1.0, shape=())
        self.build()

    def build(self):
        raise ValueError("The build() method must be implemented")

    def conv3d(self, name, dim, w_name, b_name, scope='var_name'):
        """Adds a 3D convolutional neural network layer to the model.

        Args:
            name:   A string of the name of the newly created layer.
            dim:    A tuple of the dimensions of the 3D convolutional neural network.
            w_name: A string of the TensorFlow Variable for the weights
            b_name: A string of the TensorFlow Variable for the biases
            scope:  A string of the variable scope. "var_name" refers to the scope of the weights/biases in the
                    pre-trained model.
        Returns:
            The new TensorFlow model that is generated after adding this layer.
        """
        with tf.variable_scope(scope) as var_scope:
            W = tf.get_variable(name=w_name, shape=dim, initializer=self.initializer, dtype=tf.float32)
            b = tf.get_variable(name=b_name, shape=dim[-1], initializer=tf.zeros_initializer(), dtype=tf.float32)
            tf.add_to_collection(tf.GraphKeys.WEIGHTS, W)
            tf.add_to_collection(tf.GraphKeys.BIASES, b)
        self.net = tf.nn.conv3d(self.net, W, strides=[1, 1, 1, 1, 1], padding="SAME", name=name)
        self.net = tf.nn.relu(tf.nn.bias_add(self.net, b))
        return self.net

    def maxpool(self, name, dim):
        """Adds a max pooling layer to the model.

        Args:
            name:   A string of the name of the newly created layer.
            dim:    A tuple of the dimensions of the layer.

        Returns:
            The new TensorFlow model that is generated after adding this layer.
        """
        self.net = tf.nn.max_pool3d(self.net, ksize=dim, strides=dim, padding="SAME", name=name)
        return self.net

    def fc(self, name, dim, w_name, b_name, activation = True, scope='var_name'):
        """Adds a fully connected layer to the model.

        Args:
            name:   A string of the name of the newly created layer.
            dim:    A list of the dimensions of the fully connected layer.
            w_name: A string of the TensorFlow Variable for the weights
            b_name: A string of the TensorFlow Variable for the biases
            scope:  A string of the variable scope. "var_name" refers to the scope of the weights/biases in the
                    pre-trained model.

        Returns:
            The new TensorFlow model that is generated after adding this layer.
        """
        with tf.variable_scope(scope) as var_scope:
            W = tf.get_variable(name=w_name, shape=dim, initializer=self.initializer, dtype=tf.float32)
            b = tf.get_variable(name=b_name, shape=dim[-1], initializer=tf.zeros_initializer(), dtype=tf.float32)
            tf.add_to_collection(tf.GraphKeys.WEIGHTS, W)
            tf.add_to_collection(tf.GraphKeys.BIASES, b)

        if activation:
            self.net = tf.nn.relu(tf.nn.bias_add(tf.matmul(self.net, W, name=name), b))
        else:
            self.net = tf.nn.bias_add(tf.matmul(self.net, W, name=name), b)
        return self.net

    def bc(self, name, inputs, w_name, b_name, scope='var_name'):
        """Adds a binary classifier layer to the model.

        This method simply adds a single neural net layer that outputs one value, between 0 and 1. Sigmoid is used as
        the activation of this layer.

        Args:
            name:   A string of the name of the newly created layer.
            inputs: An integer of the number of inputs this layer must take in.
            w_name: A string of the TensorFlow Variable for the weights
            b_name: A string of the TensorFlow Variable for the biases
            scope:  A string of the variable scope. "var_name" refers to the scope of the weights/biases in the
                    pre-trained model.

        Returns:
            The new TensorFlow model that is generated after adding this layer.
        """
        with tf.variable_scope(scope) as var_scope:
            W = tf.get_variable(name=w_name, shape=[inputs, 1], initializer=self.initializer, dtype=tf.float32)
            b = tf.get_variable(name=b_name, shape=1, initializer=tf.zeros_initializer(), dtype=tf.float32)
            tf.add_to_collection(tf.GraphKeys.WEIGHTS, W)
            tf.add_to_collection(tf.GraphKeys.BIASES, b)
        self.net = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(self.net, W), b))
        self.net = tf.add(self.net, 0, name=name)
        return self.net

    def reshape(self, dim):
        """Reshapes the most recent layer's output tensor.

        Args:
            dim:    A list used to reshape the network. See TensorFlow documentation for more information.

        Returns:
            The new TensorFlow model that is generated after adding this layer.
        """
        self.net = tf.reshape(self.net, dim)
        return self.net

    def dropout(self, name, prob):
        """Adds a dropout layer to the neural network.

        Args:
            name:   A string of the name of the newly created layer.
            prob:   A float value of the desired probability.

        Returns:
            The new TensorFlow model that is generated after adding this layer.
        """
        self.net = tf.nn.dropout(self.net, prob, name=name)
        return self.net
