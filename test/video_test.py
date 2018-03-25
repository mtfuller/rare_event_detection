import unittest

import shutil
from video import Video
import numpy as np

TEST_VIDEO_FILE = "test.avi"
TEST_VIDEO_HEIGHT = 240
TEST_VIDEO_WIDTH = 320
TEST_VIDEO_LENGTH = 159
TEST_VIDEO_IS_ANOMOLY = True

class TestVideo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.video = Video(TEST_VIDEO_FILE, TEST_VIDEO_IS_ANOMOLY)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree('./dataset/test')

    def test_getFrames(self):
        frames = np.array(self.__class__.video.getFrames())
        self.assertTupleEqual(frames.shape, (TEST_VIDEO_LENGTH, TEST_VIDEO_HEIGHT, TEST_VIDEO_WIDTH, 3))

    def test_isAnomaly(self):
        self.assertEqual(self.__class__.video.getAnomaly(), TEST_VIDEO_IS_ANOMOLY)

    def test_getFilename(self):
        self.assertEqual(self.__class__.video.getFilename(), TEST_VIDEO_FILE, "Did not return correct filename.")

    def test_resize(self):
        NEW_HEIGHT = 95
        NEW_WIDTH = 120
        self.video.resize(NEW_WIDTH, NEW_HEIGHT)
        frames = np.array(self.__class__.video.getFrames())
        self.assertTupleEqual(frames.shape, (TEST_VIDEO_LENGTH, NEW_HEIGHT, NEW_WIDTH, 3))

