from algorithm import Algorithm
from _dataset import dataset
TOTAL_EPOCHS = 1000

def main():
    # Load dataset
    ds = dataset()

    # Initialize algorithm
    algorithm = Algorithm(dataset=ds, model_dir="./models/UCF_Train")

    # Train the model
    print("Training model...")
    cost_curve, time = algorithm.train(total_epoch=TOTAL_EPOCHS, save_interval=200)
    print("Done. Elapsed Time: %.5f sec" % (time))

    # Test the model and display results
    print("Testing trained model...")
    fpr, tpr, auc2, time = algorithm.test()
    print("FPR: " + str(fpr))
    print("TPR: " + str(tpr))
    print("Done. AUC:%f Elapsed Time: %.5f sec" % (auc2, time))

    print("After %d iterations, the model was able to have an AUC of %f." % (TOTAL_EPOCHS, auc2))

if __name__ == "__main__":
    main()
