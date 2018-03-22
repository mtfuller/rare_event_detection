import os
import cv2



class Ucsd_folder(object):

    def __init__(self,folderName,isAnomaly):
        if folderName is None:
            raise ValueError("invalid filename argument")
        self.isAnomaly = isAnomaly
        self.folderName = folderName
        self.frames = []
        self.frameCount = 0
        self.__setFrames()


    def __setFrames(self):

        """Sets frames from the video.
                Retrieves frames from the video and returns an image array.
                Image size is 240x320
                Before running, video should be in 30FPS.
                Args:
                Returns:

                Raises:
                    ValueError: Video File Not Found.
        """

        dir = self.folderName
        if os.path.isdir(dir) is False:
            raise ValueError("Path Doesn't Exist")
        count = len(os.listdir(dir)) #count list of files inside directory

        imageArr=[]
        for i in range(count):
            frame="%02d" % i
            image = cv2.imread(dir+"/"+frame+".tif")
            if image is not None:
                imageArr.append(image)

        self.frames=imageArr
        self.frameCount = len(imageArr)

    def getFrames(self):
        return self.frames

    def getAnomaly(self):
        return self.isAnomaly

    def getFoldername(self):
        return self.folderName

    def getFrameCount(self):
        return self.frameCount

if __name__ == "__main__":
    obj = Ucsd_folder("dataset/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test/Test001",0)
    print(obj.getFrameCount())

