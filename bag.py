#author: Karim
#date: 03/13/2018

"""
When initialized, get the bag with images in each segment
Attributes:
    bag: 2D array with each row containing image array for that segment.
    video: video object to split up
    split: number of segments
"""

from video import Video
import numpy as np
import matplotlib.pyplot as plt

class Bag(object):

    def __init__(self,video,split):
        self.bag=[]
        self.video=video
        self.split=split
        self.__split_data()


    def __split_data(self):
        """Splits the video data into frames
        """
        self.bag = []
        vid=self.video
        framesCount=vid.getFrameCount()
        frames = vid.getFrames()

        addedFramesCount=0
        framesInEachSection = int (framesCount/self.split)
        for i in range(self.split):
            frameArr=[]
            for j in range(addedFramesCount,addedFramesCount+framesInEachSection):
                frameArr.append(frames[j])
            self.bag.append(frameArr)
            addedFramesCount=addedFramesCount+framesInEachSection



    def getVideo(self):
        """Return video object
                Args:
                    None
                Returns:
                    video object
                Raises:
                    None
        """
        return self.video

    def getSplitCount(self):
        """Return segments
                Args:
                    None
                Returns:
                    segment count
                Raises:
                    None
        """
        return self.split

    def getBag(self):
        """Return 2D array with segments in row and image frames in column
                Args:
                    None
                Returns:
                    2D array
                Raises:
                    None
        """
        return self.bag

    def getSegments(self):
        """Returns a collection of 16-frame segments from the video
        """
        count = self.video.getFrameCount()
        frames = np.array(self.video.getFrames()[:-(count%16)])
        segments = frames.reshape([-1, 16, 112, 112, 3])
        return segments

    def resize(self, width, height):
        """Resizes the video to the new width and height
        """
        self.video.resize(width, height)
        self.__split_data()