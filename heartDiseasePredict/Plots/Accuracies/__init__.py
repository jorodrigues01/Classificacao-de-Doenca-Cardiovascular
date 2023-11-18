import numpy as np
import matplotlib.pyplot as plt

def addLabels(x, y, color, fontsize, gap=0):
    for i in x:
        plt.text(i, y[i]+gap, y[i], ha = 'center',
                 bbox = dict(facecolor = color), size=fontsize)


def knnScores(knn_accuracies):
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(knn_accuracies, '-o', c='darkblue')
    ax.xaxis.set_major_locator(plt.FixedLocator(range(1, 16, 2)))

    addLabels(knn_accuracies.index, knn_accuracies, 'lightblue', 8, 0.2)

    ax.set_xlim((-1, 17))

    plt.title("kNN accuracy score comparison by k", {'fontsize': 20})

    plt.show()


def classifiersAccuracies(LR_score, DT_score, KNN_score):
    algorithm_accuracies = np.array([float("{:.4f}".format(LR_score)),
                                     float("{:.4f}".format(DT_score)),
                                     float("{:.4f}".format(KNN_score))])

    algorithm_accuracies *= 100

    fig, ax = plt.subplots(figsize=(8, 6))

    cmap = plt.get_cmap('Oranges')
    colors = cmap([0.3, 0.6, 0.9])

    ax.bar([0, 1, 2], algorithm_accuracies, color=colors)

    addLabels([0, 1, 2], algorithm_accuracies, 'cornsilk', 10)

    ax.xaxis.set_major_locator(plt.FixedLocator([0, 1, 2]))
    ax.xaxis.set_major_formatter(plt.FixedFormatter(['Logistic Regression', 'Decision Tree', 'KNN']))
    plt.title("Classifier's accuracy score comparison", {'fontsize': 20})

    plt.show()