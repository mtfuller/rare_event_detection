# author: Karim
# date: 03/04/2018

"""
When initialized, set the training and testing data from the ucsd file.
Attributes:
    training: array type containing training data with each row consist of video object
    testing: array type containing testing data with each row consist of video object
    video: array type containing data with each row consist of video object
    ped: use pedestrian 1 or 2 dataset file
"""
from video import Video
import os
import csv
from random import shuffle
from ucsd_folder import Ucsd_folder
import matplotlib.pyplot as plt



class ucsd_dataset(object):

    def __init__(self,**kwargs):

        self.training = []
        self.testing = []
        self.video = []
        ped=str(kwargs.get('pedestrian'))
        self.ped=ped
        self.__set_data()

    def addVideo(self, input, isAnomaly):

        filename = "dataset/ucsd_ped"+self.ped+".csv"
        if (int(isAnomaly)==1 or int(isAnomaly) == 0) is False:
            print(isAnomaly)
            raise ValueError("Invalid Anomaly Status")
        if os.path.exists(filename) is False:
            with open(filename, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(["Video Location", "is Anomaly"])

        if os.path.isdir(input) is False:
            raise ValueError("Directory doesn't exist")
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if input == row[0]:
                    raise ValueError("Entry already exist")

        with open(filename, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([input, isAnomaly])
            print("Added")

    def __set_data(self):
        row_list = []
        self.video = []
        self.training = []
        self.testing = []
        filename = "dataset/ucsd_ped" + self.ped + ".csv"
        if os.path.exists(filename) is True:
            with open(filename, 'r') as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    row_list.append(row)

            for i in range(len(row_list)):
                dir = row_list[i][0]
                anomaly = row_list[i][1]
                vid = Ucsd_folder(dir, anomaly)
                self.video.append(vid)

            # numpy.random.shuffle(self.video)
            training = []
            train_length = int(len(self.video) * .8)

            for i in range(train_length):
                training.append(self.video[i])
            self.training = training
            testing = []
            for i in range(train_length, len(row_list)):
                testing.append(self.video[i])
            self.testing = testing

    def getTraining(self):
        """return training data
                       returns the training data array with each index containing video object
                       Args:
                           None
                       Returns:
                           returns training array with video object
                       Raises:
                           None
               """
        return self.training

    def getTesting(self):
        """return testing data
                returns the testing data array with each index containing video object
                Args:
                    None
                Returns:
                    returns testing array with video object
                Raises:
                    None
        """
        return self.testing

    def getVideos(self):
        """return video data
                returns the data array with each index containing video object
                Args:
                    None
                Returns:
                    returns video object array
                Raises:
                    None
        """
        return self.video
    def randomize_data(self):

        filename = "dataset/ucsd_ped" + self.ped + ".csv"
        if os.path.exists(filename) is False:
            raise ValueError("dataset doesn't exist")
        row_list=[]
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                row_list.append(row)
        shuffle(row_list)
        with open(filename, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(["Video Location", "is Anomaly"])
            for i in range(len(row_list)):
                writer.writerow(row_list[i])

        self.__set_data()


# if __name__ == "__main__":
#
#     ped="2"
#     ucsd = ucsd_dataset(pedestrian=ped)
#     # videoName = ""
#     # while videoName!="-1":
#     #     videoName = input("Enter folder name, enter -1 to exit: ")
#     #     anomalyStatus = input("Enter anomaly status(0/1): ")
#     #     videoLoc="dataset/UCSD_Anomaly_Dataset.v1p2/UCSDped"+ped+"/Test/"+videoName+"/"
#     #     if videoName!="-1":
#     #         ucsd.addVideo(videoLoc,anomalyStatus)
#     # print("Exited")
#     # train = ucsd.getTraining()
#     #ucsd.randomize_data()
#     # print(train[0].getFrameCount())
#     # for i in range(12,13):
#     #     videoName = "Test0"+str(i)
#     #     videoName = "dataset/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test/" + videoName + "/"
#     #     ucsd.addVideo(videoName, "0")
#     arr = ucsd.getTraining()
#     sample = arr[0].getFrames()
#     image = sample[0]
#     plt.imshow(image)
#     plt.show()



