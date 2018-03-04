#author: Karim
#date: 03/04/2018

#Description: Takes a string argument for filename and split the video into frames, default directory is dataset/videos/filename
#requires cv2

#before running it, make sure that the video fps is 30, https://askubuntu.com/questions/370692/how-to-change-the-framerate-of-a-video-without-reencoding


import cv2
import os
import matplotlib.pyplot as plt

class Video(object):

    def __init__(self, filename):
        if filename is None:
            raise ValueError("invalid filename argument")

        self.filename = filename


    def __split(self):
        videoSource = cv2.VideoCapture('dataset/videos/'+self.filename)
        success,image = videoSource.read() #success is true if file exist, and loads first frame in image
        count = 0
        dir = os.path.splitext("dataset/" + self.filename)[0] #gets the directory location
        if os.path.isdir(dir) is False:
            os.makedirs(dir)
        while success:
            resize_image = cv2.resize(image,(240,320)) #240 width, 320 height
            cv2.imwrite(dir+"/frame%d.jpg" %count,resize_image)
            success, image = videoSource.read()
            count+=1
        print("Read %d Frames" %count)
        return count


    def getFrames(self):

        dir = os.path.splitext("dataset/" + self.filename)[0]  # gets the directory location
        if os.path.isdir(dir) is False:
            self.__split()

        count = len(os.listdir(dir)) #count list of files inside directory

        imageArr=[]
        for i in range(count):
            image = cv2.imread(dir+"/frame%d.jpg"%i)
            if image is not None:
                imageArr.append(image)

        return imageArr

if __name__ == "__main__":

     vid = Video("big_buck_bunny_720p_5mb.mp4")
     arr = vid.getFrames()
     print(len(arr))
     image = arr[0]
     plt.imshow(image)
     plt.show()
