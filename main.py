from algorithm import Algorithm
from ucsd_dataset import ucsd_dataset

TOTAL_EPOCHS = 3000

def main():
    ped1 = ucsd_dataset(pedestrian="1")

    algorithm = Algorithm(dataset=ped1, model_dir="./models/Ped1_Model")

    print("Testing initial model...")
    fpr, tpr, auc1, time = algorithm.test()
    print("FPR: " + str(fpr))
    print("TPR: " + str(tpr))
    print("Done. AUC:%f Elapsed Time: %.5f sec" % (auc1, time))

    print("Training model...")
    cost_curve, time = algorithm.train(total_epoch=TOTAL_EPOCHS, save_interval=500)
    print("Done. Elapsed Time: %.5f sec" % (time))

    print("Testing trained model...")
    fpr, tpr, auc2, time = algorithm.test()
    print("FPR: " + str(fpr))
    print("TPR: " + str(tpr))
    print("Done. AUC:%f Elapsed Time: %.5f sec" % (auc2, time))

    print("After %d iterations, the model was able to go from %f to %f." % (TOTAL_EPOCHS, auc1, auc2))

    #print("COST CURVE: "+str(cost_curve))

if __name__ == "__main__":
    main()
