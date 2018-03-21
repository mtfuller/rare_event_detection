import unittest

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
        myNewModel.build()

    def test_build(self):
        input_shape = (None, SEGMENT_FRAMES, SEGMENT_WIDTH, SEGMENT_HEIGHT, SEGMENT_CHANNELS)
        model_input = tuple(myNewModel.inputs.get_shape().as_list())
        self.assertTupleEqual(model_input, input_shape)

        output_shape = (None, 1)
        print(myNewModel.net.shape)
        model_input = tuple(myNewModel.net.get_shape().as_list())
        self.assertTupleEqual(model_input, output_shape)

    def test_predict(self):
        sample_segment = np.zeros((1,SEGMENT_FRAMES, SEGMENT_WIDTH, SEGMENT_HEIGHT, SEGMENT_CHANNELS))
        score = myNewModel.predict(sample_segment)
        self.assertTupleEqual(score.shape, (1,1))
        self.assertTrue(0.0 <= score[0] and score[0] <= 1.0, "The score %.4f must be between 0 and 1." % (score[0]))