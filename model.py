#author: Thomas Fuller
#date: 03/06/2018

import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np
import cv2
import matplotlib.pyplot as plt

video = cv2.VideoCapture("./vid/test.avi")

ret, frame = video.read()
st_video = np.array([frame])
print("LOADING VIDEO...")
count = 0
while video.isOpened():
    count += 1
    ret, frame = video.read()
    if not ret or count >= 32:
        break
    st_video = np.vstack((st_video, [frame]))
print("LOADED")
print(st_video.shape)

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

    def __init__(self, batch_size = 3, **kwargs):
        self.__graph = tf.Graph()
        self.__CLIP_LENGTH = kwargs.get('clip_length',32)
        self.__CROP_WIDTH = kwargs.get('crop_width', 320)
        self.__CROP_HEIGHT = kwargs.get('crop_height', 240)
        self.dropout_prob = kwargs.get('dropout_prob', 0.6)
        self.num_class = kwargs.get('num_class', 101)
        self.n_step_epoch = int(9537 / batch_size)

        with self.__graph.as_default():
            self.inputs = tf.placeholder(tf.float32, [None, self.__CLIP_LENGTH, self.__CROP_HEIGHT, self.__CROP_WIDTH, 3])
            self.__nn = self.inputs
            self.initializer = layers.xavier_initializer()
            self.global_step = tf.Variable(0, trainable=False, name="global_step")
            self.lr = tf.train.exponential_decay(1e-4, self.global_step, int(10 * self.n_step_epoch), 1e-1, True)

    def run(self):
        config = tf.ConfigProto()
        with tf.Session(config=config, graph=self.__graph) as sess:
            self.__build()

            softmax_logits = tf.nn.softmax(self.__nn)
            int_label = tf.placeholder(tf.int64, [3,])

            task_loss = tf.reduce_sum(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.__nn, labels=int_label))
            acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(softmax_logits, axis=-1), int_label), tf.float32))
            right_count = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(softmax_logits, axis=1), int_label), tf.int32))

            reg_loss = layers.apply_regularization(layers.l2_regularizer(5e-4),
                                                   tf.get_collection(tf.GraphKeys.WEIGHTS))
            total_loss = task_loss + reg_loss
            # train_var_list = [v for v in tf.trainable_variables() if v.name.find("conv") == -1]
            train_op = tf.train.GradientDescentOptimizer(self.lr).minimize(
                total_loss, global_step=self.global_step)
            # train_op = tf.train.MomentumOptimizer(self.lr,0.9).minimize(
            #     total_loss, global_step = self.global_step,var_list=train_var_list)

            total_para = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
            print('total_para:', total_para)  # all CDC9 :28613120  #pool5 27655936

            init = tf.global_variables_initializer()
            sess.run(init)
            saver = tf.train.Saver(tf.trainable_variables())
            saver.restore(sess, PRETRAINED_MODEL_PATH)
            print("Model Loading Done!")
            output = sess.run(self.__nn, feed_dict={self.inputs: [st_video]})
            print("FINISHED!!!")
            print(output)
            output = output.T
            #assert output.shape == (101,10)
            scores = np.mean(output, axis=1)
            print(scores.argsort()[-10:][::-1])

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

        # Loss Function MIL
        # L = l(B_a, B_n) + NORM(M)
        # l(B_a, B_n) = max(0,1-y_pred_a+y_pred_n) + SUM((y_pred_a - y_pred_n)**2) + SUM(y_pred_a)
        #_max = tf.maximum(np.zeros(), np.ones() - y_pred_a + y_pred_n)
        #_smooth = tf.
        #_sparsity = tf.reduce_sum(y_pred_a)
        # loss = _max + _smooth + _sparsity

    def __conv3d(self, name, dim, w_name, b_name):
        with self.__graph.as_default():
            with tf.variable_scope('var_name') as var_scope:
                W = tf.get_variable(name=w_name, shape=dim, initializer=self.initializer, dtype=tf.float32)
                b = tf.get_variable(name=b_name, shape=dim[-1], initializer=tf.zeros_initializer(), dtype=tf.float32)
                tf.add_to_collection(tf.GraphKeys.WEIGHTS, W)
                tf.add_to_collection(tf.GraphKeys.BIASES, b)
            self.__nn = tf.nn.conv3d(self.__nn, W, strides=[1, 1, 1, 1, 1], padding="SAME", name=name)
            self.__nn = tf.nn.relu(tf.nn.bias_add(self.__nn, b))

    def __maxpool(self, name, dim):
        with self.__graph.as_default():
            self.__nn = tf.nn.max_pool3d(self.__nn, ksize=dim, strides=dim, padding="SAME", name=name)

    def __fc(self, name, dim, w_name, b_name, activation = True):
        with self.__graph.as_default():
            with tf.variable_scope('var_name') as var_scope:
                W = tf.get_variable(name=w_name, shape=dim, initializer=self.initializer, dtype=tf.float32)
                b = tf.get_variable(name=b_name, shape=dim[-1], initializer=tf.zeros_initializer(), dtype=tf.float32)
                tf.add_to_collection(tf.GraphKeys.WEIGHTS, W)
                tf.add_to_collection(tf.GraphKeys.BIASES, b)

            if activation:
                self.__nn = tf.nn.bias_add(tf.matmul(self.__nn, W, name=name), b)
            else:
                self.__nn = tf.nn.relu(tf.nn.bias_add(tf.matmul(self.__nn, W, name=name), b))

    def __reshape(self, dim):
        self.__nn = tf.reshape(self.__nn, dim)

    def __dropout(self, name, prob):
        with self.__graph.as_default():
            self.__nn = tf.nn.dropout(self.__nn, prob, name=name)

