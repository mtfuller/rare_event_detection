import unittest

from tensorflow import Dimension

from model import Model
import numpy as np

SEGMENT_WIDTH = 112
SEGMENT_HEIGHT = 112
SEGMENT_FRAMES = 10
SEGMENT_CHANNELS = 3

myNewModel = Model()

class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass
        #myNewModel.build()

    def test_build(self):
        input_shape = (None, SEGMENT_FRAMES, SEGMENT_WIDTH, SEGMENT_HEIGHT, SEGMENT_CHANNELS)
        model_input = tuple(myNewModel.inputs.get_shape().as_list())
        self.assertTupleEqual(model_input, input_shape)

        # output_shape = (None, 1)
        # print(myNewModel.cnn_model.shape)
        # model_input = tuple(myNewModel.cnn_model.get_shape().as_list())
        # self.assertTupleEqual(model_input, output_shape)

    def test_predict(self):
        sample_segment = np.zeros((SEGMENT_FRAMES, SEGMENT_WIDTH, SEGMENT_HEIGHT, SEGMENT_CHANNELS))
        score = myNewModel.predict(sample_segment)
        self.assertTupleEqual(score.shape, (1,))