#author: Thomas Fuller
#date: 03/06/2018

import tensorflow as tf

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
        likes_spam: A boolean indicating if we like SPAM or not.
        eggs: An integer count of the eggs we have laid.
    """

    def __init__(self, **kwargs):
        self.__graph = tf.Graph()
        self.__CLIP_LENGTH = kwargs.get('clip_length',10)
        self.__CROP_WIDTH = kwargs.get('crop_width', 10)
        self.__CROP_HEIGHT = kwargs.get('crop_height', 10)
        self.dropout_prob = kwargs.get('dropout_prob', 0.6)
        self.num_class = kwargs.get('num_class', 101)

        with self.__graph.as_default():
            self.inputs = tf.placeholder(tf.float32, [None, self.__CLIP_LENGTH, self.__CROP_HEIGHT, self.__CROP_WIDTH, 3])
            self.initializer = tf.contrib.xavier_initializer()

    def saveModel(self):
        pass

    def loadModel(self):
        pass

    def __build(self):
        # The C3D CNN Architecture
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
        self.__dropout('dropout1', self.dropout_prob)
        self.__fc('fc2', [4096, 4096], 'wd2', 'bd2')
        self.__dropout('dropout2', self.dropout_prob)
        self.__fc('fc3', [4096, self.num_class], 'wout', 'bout', False)

    def __conv3d(self, name, dim, w_name, b_name):
        with self.__graph.as_default():
            with tf.variable_scope('var_name') as var_scope:
                W = tf.get_variable(name=w_name, shape=dim, initializer=self.initializer, dtype=tf.float32)
                b = tf.get_variable(name=b_name, shape=dim[-1], initializer=tf.zeros_initializer(), dtype=tf.float32)
                tf.add_to_collection(tf.GraphKeys.WEIGHTS, W)
                tf.add_to_collection(tf.GraphKeys.BIASES, b)
            self.__model = tf.nn.conv3d(self.__model, W, strides=[1, 1, 1, 1, 1], padding="SAME", name=name)
            self.__model = tf.nn.relu(tf.nn.bias_add(self.__model, b))

    def __maxpool(self, name, dim):
        with self.__graph.as_default():
            self.__model = tf.nn.max_pool3d(self.__model, ksize=dim, strides=dim, padding="SAME", name=name)

    def __fc(self, name, dim, w_name, b_name, activation = True):
        with self.__graph.as_default():
            with tf.variable_scope('var_name') as var_scope:
                W = tf.get_variable(name=w_name, shape=dim, initializer=self.initializer, dtype=tf.float32)
                b = tf.get_variable(name=b_name, shape=dim[-1], initializer=tf.zeros_initializer(), dtype=tf.float32)
                tf.add_to_collection(tf.GraphKeys.WEIGHTS, W)
                tf.add_to_collection(tf.GraphKeys.BIASES, b)

            if activation:
                self.__model = tf.nn.bias_add(tf.matmul(self.__model, W, name=name), b)
            else:
                self.__model = tf.nn.relu(tf.nn.bias_add(tf.matmul(self.__model, W, name=name), b))

    def __reshape(self, dim):
        self.__model = tf.reshape(self.__model, dim)

    def __dropout(self, name, prob):
        with self.__graph.as_default():
            self.__model = tf.nn.dropout(self.__model, prob, name=name)

