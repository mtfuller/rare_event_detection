#author: Thomas Fuller
#date: 03/06/2018

import tensorflow as tf
import numpy as np
import time
from tensorflow.contrib import layers

PRETRAINED_MODEL_PATH = "./pretrained_models/c3d_ucf101_finetune_whole_iter_20000_TF.model"

class Model:
    """A modified implementation of the C3D architecture, designed for binary classification.

    This class follows closely to the C3D-TensorFlow project, a TensorFlow implementation of the Caffe C3D model. The
    major difference is that this model follows an architecture designed for anomaly detection in videos. You can read
    more about the architecture at: http://crcv.ucf.edu/cchen/. This class allows a C3D model to be built from scratch,
    setting up the layers and loading pre-trained parameters. Alternatively, models can be loaded straight from a file
    in the working directory. For more information on the TensorFlow implementation, please visit:
    https://github.com/hx173149/C3D-tensorflow.

    Attributes:
        session:    The active TensorFlow session.
        net:        The generated TensorFlow neural network we are using for training, testing, and evalutating.
    """
    def __init__(self, **kwargs):
        """Initializes the model and configures TensorFlow session"""
        self.__graph = tf.Graph()
        self.__CLIP_LENGTH = kwargs.get('clip_length',4)
        self.__CROP_WIDTH = kwargs.get('crop_width', 112)
        self.__CROP_HEIGHT = kwargs.get('crop_height', 112)
        self.__DROPOUT_PROB = kwargs.get('dropout_prob', 0.6)
        self.__NET_LAYERS = kwargs.get('net_layers', ['conv1', 'conv2', 'conv3a', 'conv3b', 'conv4a', 'conv4b',
                                                      'conv5a', 'conv5b','fc1', 'fc4', 'fc5', 'fc6'])
        config = tf.ConfigProto()
        self.session = tf.Session(config=config, graph=self.__graph)

    def predict(self, video_segments):
        """Returns an anomaly score based on the given set of video segment.

        Runs a forward propagation on the network, which computes an anomaly score between 0 and 1.

        Args:
            video_segments: A 5-dimensional Numpy array that has a shape that corresponds to the following: (segments,
                            frames, width, height, channels). For example, if the method was given 3 video segments,
                            each with 10 frames, 112x112 RGB resolution, the video segment shape would be:
                            (3, 10, 112, 112, 3).

        Returns:
            A numpy array of float values that has a size equal to the given number of segments. For example, if a set
            of 3 video segments was passed in, the function would return an array of size 3, where each element is an
            anomaly score of the corresponding video segment.
        """
        with self.__graph.as_default():
            start = time.time()
            output = self.session.run(self.net, feed_dict={self.inputs: video_segments})
            end = time.time()
            return output, (end-start)

    def train(self, positive_bag, negative_bag):
        with self.__graph.as_default():
            frames = np.vstack((positive_bag, negative_bag))
            _, c = self.session.run([self.optimizer, self.loss], feed_dict={
                self.inputs: frames,
                self.prob: self.__DROPOUT_PROB
            })
            return c

    def saveModel(self, export_dir):
        with self.__graph.as_default():
            saver0 = tf.train.Saver()
            saver0.save(self.session, export_dir)
            saver0.export_meta_graph(export_dir + '.meta')

    def loadModel(self, export_dir):
        with self.__graph.as_default():
            new_saver = tf.train.import_meta_graph(export_dir + '.meta')
            new_saver.restore(self.session, export_dir)
            self.inputs = self.__graph.get_tensor_by_name("Placeholder:0")
            self.net = self.__graph.get_tensor_by_name("fc6:0")

    def build(self):
        """Constructs the CNN architecture for the model we are evaluating.

        This method initializes a TensorFlow graph, loads in the pre-trained C3D network, and finally appends a 3 layer
        neural net to compute an anomaly score. For more information about the architecture please visit:
        http://crcv.ucf.edu/cchen/.
        """
        with self.__graph.as_default():
            # Initializes and sets up the inputs, weight/bias initializer, and other global variables
            self.inputs = tf.placeholder(tf.float32, [None,
                                                      self.__CLIP_LENGTH,
                                                      self.__CROP_HEIGHT,
                                                      self.__CROP_WIDTH,
                                                      3])
            self.net = self.inputs
            self.initializer = layers.xavier_initializer()
            self.global_step = tf.Variable(0, trainable=False, name="global_step")

            self.prob = tf.placeholder_with_default(1.0, shape=())

            # Constructs the C3D network, based on the C3D-Tensorflow implementation of the original model written in
            # Caffe.
            self.__conv3d('conv1', [3, 3, 3, 3, 64], 'wc1', 'bc1')
            self.__maxpool('pool1', [1, 1, 2, 2, 1])
            self.__conv3d('conv2', [3, 3, 3, 64, 128], 'wc2', 'bc2')
            self.__maxpool('pool2', [1, 2, 2, 2, 1])
            self.__conv3d('conv3a', [3, 3, 3, 128, 256], 'wc3a', 'bc3a')
            self.__conv3d('conv3b', [3, 3, 3, 256, 256], 'wc3b', 'bc3b')
            self.__maxpool('pool3', [1, 2, 2, 2, 1])
            self.__conv3d('conv4a', [3, 3, 3, 256, 512], 'wc4a', 'bc4a')
            self.__conv3d('conv4b', [3, 3, 3, 512, 512], 'wc4b', 'bc4b')
            self.__maxpool('pool4', [1, 2, 2, 2, 1])
            self.__conv3d('conv5a', [3, 3, 3, 512, 512], 'wc5a', 'bc5a')
            self.__conv3d('conv5b', [3, 3, 3, 512, 512], 'wc5b', 'bc5b')
            self.__maxpool('pool5', [1, 2, 2, 2, 1])
            self.__reshape([-1, 8192])
            self.__fc('fc1', [8192, 4096], 'wd1', 'bd1')
            fc6_layer = self.__dropout('dropout1', self.prob)
            self.__fc('fc2', [4096, 4096], 'wd2', 'bd2')
            self.__dropout('dropout2', self.prob)
            self.c3d_output = self.__fc('fc3', [4096, 101], 'wout', 'bout', False)

            # Initializes all of the weights and biases created so far
            init = tf.global_variables_initializer()
            self.session.run(init)

            # Loads in the pre-trained C3D model
            saver = tf.train.Saver(tf.trainable_variables())
            saver.restore(self.session, PRETRAINED_MODEL_PATH)

            # Append the anomaly score 3 layer neural net
            self.net = fc6_layer
            self.__fc('fc4', [4096, 512], 'wd4', 'bd4', scope="test123")
            self.__dropout('dropout3', self.prob)
            self.__fc('fc5', [512, 32], 'wd5', 'bd5', scope="test123")
            self.__dropout('dropout3', self.prob)
            self.__bc('fc6', 32, 'wd6', 'bd6', scope="test123")

            with tf.variable_scope('test123') as var_scope:

                self.video_a = tf.gather(self.net, tf.range(0, 32))
                self.video_n = tf.gather(self.net, tf.range(32, 64))

                self.max_a = tf.reduce_max(self.video_a)
                self.max_n = tf.reduce_max(self.video_n)

                lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()
                                   if 'bias' not in v.name]) * 0.001

                # TODO: Look into implementing smoothness and sparcity into loss function
                self.loss = tf.reduce_mean(tf.maximum(0.0, 1 - self.max_a + self.max_n) + lossL2)

                self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)

            # Initialize the weights of the new neural net layers
            vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='test123')
            self.session.run(tf.variables_initializer(vars))




    def __conv3d(self, name, dim, w_name, b_name, scope='var_name'):
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

    def __maxpool(self, name, dim):
        """Adds a max pooling layer to the model.

        Args:
            name:   A string of the name of the newly created layer.
            dim:    A tuple of the dimensions of the layer.

        Returns:
            The new TensorFlow model that is generated after adding this layer.
        """
        self.net = tf.nn.max_pool3d(self.net, ksize=dim, strides=dim, padding="SAME", name=name)
        return self.net

    def __fc(self, name, dim, w_name, b_name, activation = True, scope='var_name'):
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

    def __bc(self, name, inputs, w_name, b_name, scope='var_name'):
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

    def __reshape(self, dim):
        """Reshapes the most recent layer's output tensor.

        Args:
            dim:    A list used to reshape the network. See TensorFlow documentation for more information.

        Returns:
            The new TensorFlow model that is generated after adding this layer.
        """
        self.net = tf.reshape(self.net, dim)
        return self.net

    def __dropout(self, name, prob):
        """Adds a dropout layer to the neural network.

        Args:
            name:   A string of the name of the newly created layer.
            prob:   A float value of the desired probability.

        Returns:
            The new TensorFlow model that is generated after adding this layer.
        """
        self.net = tf.nn.dropout(self.net, prob, name=name)
        return self.net

    def __str__(self):
        s = ""
        with self.__graph.as_default():
            for layer in self.__NET_LAYERS:
                shape = self.__graph.get_tensor_by_name(layer+":0").shape
                s += "LAYER: %s\tSHAPE: %s\n" % (layer, shape)
        return s