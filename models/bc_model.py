from models.trainable_model import TrainableModel
import tensorflow as tf

BC_SHAPE = (None, 4096)

class BCModel(TrainableModel):
    def __init__(self, **kwargs):
        super().__init__(BC_SHAPE, output_name='fc6', **kwargs)

    def build(self):
        with self.graph.as_default():
            self.reshape([-1, 4096])
            self.fc('fc4', [4096, 512], 'wd4', 'bd4')
            self.dropout('dropout3', self.prob)
            self.fc('fc5', [512, 32], 'wd5', 'bd5')
            self.dropout('dropout4', self.prob)
            self.bc('fc6', 32, 'wd6', 'bd6')

            # Segment: (Videos, Segments, Anomaly Score)
            segment = tf.reshape(self.net, [-1, 12, 1])

            # Positive Bag: (Videos, Segments, Anomaly Score)
            # Negative Bag: (Videos, Segments, Anomaly Score)
            positive_bag, negative_bag = tf.split(segment, num_or_size_splits=2)

            # Max Positive: (Videos, Anomaly Score)
            max_positive = tf.reduce_max(positive_bag, axis=1)
            # Max Negative: (Videos, Anomaly Score)
            max_negative = tf.reduce_max(negative_bag, axis=1)

            lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()
                               if 'bias' not in v.name]) * 0.001

            self.loss = tf.reduce_mean(tf.maximum(0.0, 1 - max_positive + max_negative) + lossL2)

            self.optimizer = tf.train.AdagradOptimizer(learning_rate=0.001).minimize(self.loss)

            # Initializes all of the weights and biases created so far
            init = tf.global_variables_initializer()
            self.session.run(init)
