#author: Karim
#date: 03/04/2018

#Description: Use to add/remove data from the csv file
import video
import os
import csv
import numpy

class dataset:

    def __init__(self):
        
        self.training=[]
        self.testing=[]

        self.__set_data()

    def addVideo(self,filename,isAnomaly):

        if filename is None:
            raise ValueError("invalid filename argument")
        if isAnomaly>1 or isAnomaly < 0:
            raise ValueError("invalid anomaly argument")

        vid = video.Video(filename)
        imgArr = vid.getFrames()
        dir = os.path.splitext("dataset/" + filename)[0]

        if os.path.exists("dataset/data.csv") is False:
            with open('dataset/data.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerow(["Directory", "is Anomaly"])

        with open('dataset/data.csv', 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if dir==row[0]:
                    raise ValueError("Entry already exist")


        with open('dataset/data.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([dir,isAnomaly.bit_length()])

        self.__set_data()

        return imgArr #frames

    def removeVideo(self, filename):
        if filename is None:
            raise ValueError("invalid filename argument")

        dir = os.path.splitext("dataset/" + filename)[0]

        found=False
        count=0

        row_list = []
        with open('dataset/data.csv', 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if dir==row[0]:
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
        with open('dataset/data.csv', 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                row_list.append(row)

        numpy.random.shuffle(row_list)
        training = []
        train_length = int(len(row_list)*.8)

        for i in range(train_length):
            training.append(row_list[i])
        self.training=training
        testing=[]
        for i in range(train_length,len(row_list)):
            testing.append(row_list[i])
        self.testing=testing

    def randomize_data(self):
        self.__set_data()


    def getTraining(self):
        return self.training

    def getTesting(self):
        return self.testing

if __name__ == "__main__":
    ds = dataset()
    ds.addVideo("big_buck_bunny_720p_5mb.mp4",False)
    #ds.addVideo("big_buck_bunny_720p_5mb2.mp4", 1)
    #ds.removeVideo("big_buck_bunny_720p_5mb3.mp4")