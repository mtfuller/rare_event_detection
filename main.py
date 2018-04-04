from algorithm import Algorithm
from ucsd_dataset import ucsd_dataset

def main():
    ped1 = ucsd_dataset(pedestrian="1")

    algorithm = Algorithm(dataset=ped1, model_dir="./models/Ped1_Model")

    print("Testing initial model...")
    fpr, tpr, auc1, time = algorithm.test()
    #print("LABELS: " + str(labels))
    print("AUC: " + str(auc1))
    print("Done. AUC: Elapsed Time: %.5f sec" % (time))

    print("Training model...")
    cost_curve, time = algorithm.train(total_epoch=300, save_interval=50)
    print("Done. Elapsed Time: %.5f sec" % (time))

    print("Testing trained model...")
    fpr, tpr, auc2, time = algorithm.test()
    print("Done. AUC:%f Elapsed Time: %.5f sec" % (auc2, time))

    #print("COST CURVE: "+str(cost_curve))

if __name__ == "__main__":
    main()
