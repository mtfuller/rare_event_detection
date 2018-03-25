import unittest, shutil

from bag import Bag
from model import Model
import numpy as np

from ucsd_dataset import ucsd_dataset

SEGMENT_WIDTH = 112
SEGMENT_HEIGHT = 112
SEGMENT_FRAMES = 4
SEGMENT_CHANNELS = 3

MODEL_PATH = "./models/MODEL_TEST_001"
MODEL_FILENAME = "/model"

myNewModel = Model()

class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        myNewModel.build()
        sample_segment = np.zeros((1, SEGMENT_FRAMES, SEGMENT_WIDTH, SEGMENT_HEIGHT, SEGMENT_CHANNELS))
        cls.score, time = myNewModel.predict(sample_segment)
        myNewModel.saveModel(MODEL_PATH + MODEL_FILENAME)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(MODEL_PATH)

    def test_build(self):
        input_shape = (None, SEGMENT_FRAMES, SEGMENT_WIDTH, SEGMENT_HEIGHT, SEGMENT_CHANNELS)
        model_input = tuple(myNewModel.inputs.get_shape().as_list())
        self.assertTupleEqual(model_input, input_shape)
        output_shape = (None, 1)
        model_input = tuple(myNewModel.net.get_shape().as_list())
        self.assertTupleEqual(model_input, output_shape)

    def test_predict(self):
        sample_segment = np.zeros((1,SEGMENT_FRAMES, SEGMENT_WIDTH, SEGMENT_HEIGHT, SEGMENT_CHANNELS))
        score, time = myNewModel.predict(sample_segment)
        self.assertTupleEqual(score.shape, (1,1))
        self.assertTrue(0.0 <= score[0] and score[0] <= 1.0, "The score %.4f must be between 0 and 1." % (score[0]))

    def test_save_load(self):
        sample_segment = np.zeros((1, SEGMENT_FRAMES, SEGMENT_WIDTH, SEGMENT_HEIGHT, SEGMENT_CHANNELS))
        myNewModel2 = Model()
        myNewModel2.loadModel(MODEL_PATH + MODEL_FILENAME)
        new_score, time = myNewModel2.predict(sample_segment)
        score = self.__class__.score
        self.assertEqual(new_score, score, "The loaded model predicted %f, but should have predicted %f" % (new_score, score))

    def test_train(self):
        ucsd = ucsd_dataset(pedestrian="2")
        training = ucsd.getTraining()
        positive_bag = negative_bag = None

        for video in training:
            if video.getAnomaly() == '1' and not positive_bag:
                positive_bag = Bag(video, 32)
            if video.getAnomaly() == '0'  and not negative_bag:
                negative_bag = Bag(video, 32)

        positive_bag.resize(112, 112)
        negative_bag.resize(112, 112)

        positive_bag = positive_bag.getSegments()
        negative_bag = negative_bag.getSegments()

        cost = myNewModel.train(positive_bag, negative_bag)
        self.assertTrue(isinstance(cost, np.float32) and cost > 0.0, "Returned cost value is not valid: %s" % cost)

