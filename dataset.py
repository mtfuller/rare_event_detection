#author: Karim
#date: 03/04/2018

"""
When initialized, set the training and testing data from the dataset file.
Attributes:
    training: array type containing training data with each row consist of video object
    testing: array type containing testing data with each row consist of video object
    video: array type containing data with each row consist of video object
"""
from video import Video
import os
import csv
from random import shuffle

class dataset:

    def __init__(self):
        
        self.training=[]
        self.testing=[]
        self.video=[]
        self.__set_data()

    def addVideo(self,filename,isAnomaly):

        """Add video data into dataset
                Update the csv dataset file with the new data then update the training and testing dataset
                Args:
                    filename: a string specifying video file name along with the extension
                    isAnomaly: a boolean specifying if it's anomaly, true if it's an anomaly else false
                Returns:
                    True if video is added, otherwise raises error
                Raises:
                    ValueError: File Name argument Not Specified.
                    ValueError: Anomaly argument is not boolean.
                    ValueError: Entry already exist in the dataset
        """


        if filename is None:
            raise ValueError("invalid filename argument")
        if isAnomaly>1 or isAnomaly < 0:
            raise ValueError("invalid anomaly argument")

        vid = Video(filename,isAnomaly)

        loc="dataset/videos/"+filename
        if os.path.exists("dataset/data.csv") is False:
            with open('dataset/data.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerow(["Video Location", "is Anomaly"])

        with open('dataset/data.csv', 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if loc==row[0]:
                    raise ValueError("Entry already exist")


        with open('dataset/data.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([loc,isAnomaly.bit_length()])

        self.__set_data()

        return True

    def removeVideo(self, filename):
        """remove video data from dataset
                Update the csv dataset file by removing the specified data then update the training and testing dataset
                Args:
                    filename: a string specifying video file name along with the extension
                Returns:
                    A boolean with true if removal is successful, else false
                Raises:
                    ValueError: File Name argument Not Specified.
                    ValueError: Video file not in the dataset
        """
        if os.path.exists("dataset/data.csv") is True:
            if filename is None:
                raise ValueError("invalid filename argument")

            fileLoc = "dataset/videos/"+filename

            found=False

            row_list = []
            with open('dataset/data.csv', 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if fileLoc==row[0]:
                        found=True
                        continue
                    row_list.append(row)

            with open('dataset/data.csv', 'w') as f:
                writer = csv.writer(f)
                for i in range(len(row_list)):
                    writer.writerow(row_list[i])

            if found is False:
                raise ValueError("Filename not in list")

            self.__set_data()



    def __set_data(self):
        row_list = []
        self.video=[]
        self.training=[]
        self.testing=[]
        if os.path.exists("dataset/data.csv") is True:
            with open('dataset/data.csv', 'r') as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    row_list.append(row)

            for i in range(len(row_list)):
                filename = os.path.basename(row_list[i][0])
                anomaly = os.path.basename(row_list[i][1])
                vid = Video(filename,anomaly)
                self.video.append(vid)

            #numpy.random.shuffle(self.video)
            training = []
            train_length = int(len(self.video)*.8)

            for i in range(train_length):
                training.append(self.video[i])
            self.training=training
            testing=[]
            for i in range(train_length,len(row_list)):
                testing.append(self.video[i])
            self.testing=testing

    def randomize_data(self):
        """Randomizes the training and testing data
                Randomize the data and assign 80% to training and 20% to testing
                Args:
                    None
                Returns:
                    None
                Raises:
                    None
        """
        filename = "dataset/data.csv"
        if os.path.exists(filename) is False:
            raise ValueError("dataset doesn't exist")
        row_list = []
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

if __name__ == "__main__":
    ds = dataset()
    ds.addVideo("SampleVideo_1280x720_1mb.mp4",True)
    #ds.addVideo("big_buck_bunny_720p_5mb.mp4", False)
    #ds.addVideo("SampleVideo_1280x720_2mb.mp4", False)
    #ds.removeVideo("SampleVideo_1280x720_2mb.mp4")
    #ds.removeVideo("SampleVideo_1280x720_1mb.mp4")
    #ds.removeVideo("big_buck_bunny_720p_5mb.mp4")
    training = ds.getTraining()
    testing = ds.getTesting()
    if(len(training)>0):

        vid = training[0]
        filename = vid.getFilename()
        frames = vid.getFrames()
        frameC = vid.getFrameCount()
        anomaly = vid.getAnomaly()
        print("Total Objects in Training Data: %d" %len(training))
        print("First object in Training Data")
        print("Filename: " + filename)
        print("Frames Count: %d" %frameC)
        print("Anomaly Status: " + anomaly)
        for i in range(len(training)):
            print(training[i])

    print("--- \t --- \t --- \t ---")

    if(len(testing)>0):
        vid = testing[0]
        filename = vid.getFilename()
        frames = vid.getFrames()
        frameC = vid.getFrameCount()
        anomaly = vid.getAnomaly()
        print("Total Objects in Testing Data: %d" % len(testing))
        print("First object in Testing Data")
        print("Filename: " + filename)
        print("Frames Count: %d" %frameC)
        print("Anomaly Status: " + anomaly)

        for i in range(len(testing)):
            print(testing[i])

    #ds.removeVideo("SampleVideo_1280x720_2mb.mp4")
    #ds.removeVideo("SampleVideo_1280x720_1mb.mp4")
    #ds.removeVideo("big_buck_bunny_720p_5mb.mp4")

    #vid = ds.getTraining()
    #ds.addVideo("big_buck_bunny_720p_5mb.mp4",False)
    #ds.addVideo("big_buck_bunny_720p_5mb2.mp4", 1)
