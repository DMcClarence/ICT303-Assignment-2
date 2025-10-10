import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_confusion_matrix(save_dir, est, gnd_truth, labels):
    cm = confusion_matrix(y_pred=est, y_true=gnd_truth)
    dcm = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    dcm.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.cm = cm
    plt.savefig(save_dir + "/ConfusionMatrix.png")
    plt.show()