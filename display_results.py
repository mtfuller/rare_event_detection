import matplotlib.pyplot as plt
import json

def display_results(result_dict):
    """Takes a dictionary of ROC Curve values to a matplotlib graph.
    """
    plt.title('Receiver Operating Characteristic')
    plt.plot(result_dict["fpr"], result_dict["tpr"], 'b', label='AUC = %0.3f' % result_dict["auc"])
    plt.legend(loc='lower right')
    plt.plot([0,1], [0,1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

def main():
    """Display each .json file given in the list
    """
    for i in ["UCF_AFTER_RESULTS"]:
        display_results(json.load(open('./results/' + i + ".json")))

if __name__ == '__main__':
    main()