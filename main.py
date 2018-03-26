from algorithm import Algorithm
from ucsd_dataset import ucsd_dataset

def main():
    ped1 = ucsd_dataset(pedestrian="1")

    algorithm = Algorithm(dataset=ped1, model_dir="./models/Ped1_Model")
    algorithm.build()

    print("Testing initial model...")
    labels, pred, time = algorithm.test()
    print("LABELS: " + str(labels))
    print("PREDICTIONS: " + str(labels))
    print("Done. AUC: Elapsed Time: %.5f sec" % (time))

    print("Training model...")
    cost_curve, time = algorithm.train(total_epoch=50, save_interval=5)
    print("Done. Elapsed Time: %.5f sec" % (time))

    print("Testing trained model...")
    labels, pred, time = algorithm.test()
    print("LABELS: " + str(labels))
    print("PREDICTIONS: " + str(labels))
    print("Done. AUC: Elapsed Time: %.5f sec" % (time))

    print("COST CURVE: "+str(cost_curve))

if __name__ == "__main__":
    main()