import unittest, shutil

from bag import Bag
from model import Model
import numpy as np

from models.c3d_model import C3DModel
from ucsd_dataset import ucsd_dataset

SEGMENT_WIDTH = 112
SEGMENT_HEIGHT = 112
SEGMENT_FRAMES = 16
SEGMENT_CHANNELS = 3

myNewModel = C3DModel()

class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sample_segment = np.zeros((1, SEGMENT_FRAMES, SEGMENT_WIDTH, SEGMENT_HEIGHT, SEGMENT_CHANNELS))
        cls.score, time = myNewModel.predict(sample_segment)

    def test_build(self):
        input_shape = (None, SEGMENT_FRAMES, SEGMENT_WIDTH, SEGMENT_HEIGHT, SEGMENT_CHANNELS)
        model_input = tuple(myNewModel.inputs.get_shape().as_list())
        self.assertTupleEqual(model_input, input_shape)
        output_shape = (None, 4096)
        model_input = tuple(myNewModel.net.get_shape().as_list())
        self.assertTupleEqual(model_input, output_shape)

    def test_predict(self):
        sample_segment = np.zeros((1,SEGMENT_FRAMES, SEGMENT_WIDTH, SEGMENT_HEIGHT, SEGMENT_CHANNELS))
        score, time = myNewModel.predict(sample_segment)
        self.assertTupleEqual(score.shape, (1,4096))
        self.assertTrue(score[0][0] >= 0.0 and score[0][0] <= 1.0, "The returned score was not normalizized (between 0 & 1).")