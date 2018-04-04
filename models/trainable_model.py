from models.abstract_model import AbstractModel
import tensorflow as tf
import numpy as np
import os

class TrainableModel(AbstractModel):
    def __init__(self, input_shape, **kwargs):
        super().__init__(input_shape, **kwargs)

    def save_model(self, export_dir):
        with self.graph.as_default():
            saver0 = tf.train.Saver()
            saver0.save(self.session, export_dir)

    def load_model(self, export_dir, input_name=None, output_name=None):
        input_name = self.input_name if not input_name else input_name
        output_name = self.output_name if not output_name else output_name
        with self.graph.as_default():
            saver0 = tf.train.Saver()
            saver0.restore(self.session, export_dir)
            self.inputs = self.graph.get_tensor_by_name(input_name + ":0")
            self.net = self.graph.get_tensor_by_name(output_name + ":0")

    def train(self, positive_batch, negative_batch):
        with self.graph.as_default():
            frames = np.vstack((positive_batch, negative_batch))
            _, c = self.session.run([self.optimizer, self.loss], feed_dict={
                self.inputs: frames,
                self.prob: 0.6
            })
            return c
