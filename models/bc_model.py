from models.trainable_model import TrainableModel
import numpy as np
import tensorflow as tf

BC_SHAPE = (None, 4096)

class BCModel(TrainableModel):
    def __init__(self, **kwargs):
        super().__init__(BC_SHAPE, output_name='fc6', **kwargs)

    def train(self, positive_batch, negative_batch):
        """Train the model using the positive and negative batch
        """
        with self.graph.as_default():
            print("PRINTING BATCHES: (%s, %s)" % (len(positive_batch),len(negative_batch)))
            frames = np.vstack((positive_batch, negative_batch))
            print("FRAMES SHAPE: %s" % (str(frames.shape)))
            _, c = self.session.run([self.optimizer, self.loss], feed_dict={
                self.inputs: frames,
                self.prob: 0.6,
                self.pos_neg_bag_split: (len(positive_batch)/32,len(negative_batch)/32)
            })
            return c

    def build(self):
        with self.graph.as_default():
            self.reshape([-1, 4096])
            self.fc('fc4', [4096, 512], 'wd4', 'bd4')
            self.dropout('dropout3', self.prob)
            self.fc('fc5', [512, 32], 'wd5', 'bd5')
            self.dropout('dropout4', self.prob)
            self.bc('fc6', 32, 'wd6', 'bd6')

            # Segment: (Videos, Segments, Anomaly Score)
            segment = tf.reshape(self.net, [-1, 32, 1])

            # Positive Bag: (Videos, Segments, Anomaly Score)
            # Negative Bag: (Videos, Segments, Anomaly Score)
            self.pos_neg_bag_split = tf.placeholder(tf.int32, (2))
            positive_bag, negative_bag = tf.split(segment, self.pos_neg_bag_split, axis=0)

            # Max Positive: (Videos, Anomaly Score)
            max_positive = tf.reduce_max(positive_bag)
            # Max Negative: (Videos, Anomaly Score)
            max_negative = tf.reduce_max(negative_bag)

            lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()
                               if 'bias' not in v.name]) * 0.001

            self.loss = tf.reduce_mean(tf.maximum(0.0, 1 - max_positive + max_negative) + lossL2)

            self.optimizer = tf.train.AdagradOptimizer(learning_rate=0.001).minimize(self.loss)

            # Initializes all of the weights and biases created so far
            init = tf.global_variables_initializer()
            self.session.run(init)
