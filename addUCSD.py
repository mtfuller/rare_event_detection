import os
import csv

def add_data(input,anomaly):
    filename = "dataset/ucsd_ped1.csv"
    if int(anomaly)>1 or int(anomaly)<0:
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
            if input==row[0]:
                raise ValueError("Entry already exist")

    with open(filename, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([input,anomaly])
        print("Added")

def test_dataset():
    row_list=[]
    video=[]
    filename = "dataset/ucsd_ped1.csv"
    if os.path.exists(filename) is True:
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                row_list.append(row)

        for i in range(len(row_list)):
            print(os.path.basename(row_list[i][0]))
            filename = os.path.basename(row_list[i][0])
            anomaly = os.path.basename(row_list[i][1])
            # print(filename)
            # print(anomaly)



if __name__ == "__main__":

    test_dataset()
    # for i in range(10,17):
    #     videoName = "Train0"+str(i)
    #     videoName = "dataset/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Train/" + videoName + "/"
    #     add_data(videoName, "0")