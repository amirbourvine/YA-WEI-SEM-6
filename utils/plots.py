
import matplotlib.pyplot as plt
import numpy
from sklearn import metrics


def confusion_mat(test_gt,test_preds, labels):
    confusion_matrix = metrics.confusion_matrix(test_gt,test_preds)

    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = labels)

    cm_display.plot()
    plt.show()