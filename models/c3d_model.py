from models.abstract_model import AbstractModel
import tensorflow as tf

C3D_INPUT = (None, 16, 112, 112, 3)
PRETRAINED_MODEL_PATH = "./models/pretrained_models/c3d_ucf101_finetune_whole_iter_20000_TF.model"


class C3DModel(AbstractModel):
    def __init__(self, **kwargs):
        super().__init__(C3D_INPUT, output_name="fc1", **kwargs)

    def build(self):
        # Constructs the C3D network, based on the C3D-Tensorflow implementation of the original model written in
        # Caffe.
        with self.graph.as_default():
            self.conv3d('conv1', [3, 3, 3, 3, 64], 'wc1', 'bc1')
            self.maxpool('pool1', [1, 1, 2, 2, 1])
            self.conv3d('conv2', [3, 3, 3, 64, 128], 'wc2', 'bc2')
            self.maxpool('pool2', [1, 2, 2, 2, 1])
            self.conv3d('conv3a', [3, 3, 3, 128, 256], 'wc3a', 'bc3a')
            self.conv3d('conv3b', [3, 3, 3, 256, 256], 'wc3b', 'bc3b')
            self.maxpool('pool3', [1, 2, 2, 2, 1])
            self.conv3d('conv4a', [3, 3, 3, 256, 512], 'wc4a', 'bc4a')
            self.conv3d('conv4b', [3, 3, 3, 512, 512], 'wc4b', 'bc4b')
            self.maxpool('pool4', [1, 2, 2, 2, 1])
            self.conv3d('conv5a', [3, 3, 3, 512, 512], 'wc5a', 'bc5a')
            self.conv3d('conv5b', [3, 3, 3, 512, 512], 'wc5b', 'bc5b')
            self.maxpool('pool5', [1, 2, 2, 2, 1])
            self.reshape([-1, 8192])
            final_layer = self.fc('fc1', [8192, 4096], 'wd1', 'bd1')
            self.dropout('dropout1', self.prob)
            self.fc('fc2', [4096, 4096], 'wd2', 'bd2')
            self.dropout('dropout2', self.prob)
            self.fc('fc3', [4096, 101], 'wout', 'bout', False)

            self.net = tf.nn.l2_normalize(final_layer, dim=0, name=self.output_name)

            # Initializes all of the weights and biases created so far
            init = tf.global_variables_initializer()
            self.session.run(init)

            # Loads in the pre-trained C3D model
            saver = tf.train.Saver(tf.trainable_variables())
            saver.restore(self.session, PRETRAINED_MODEL_PATH)
