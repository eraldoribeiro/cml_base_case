import json
import os

from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# Read in data
X_train = np.genfromtxt("data/train_features.csv")
y_train = np.genfromtxt("data/train_labels.csv")
X_test = np.genfromtxt("data/test_features.csv")
y_test = np.genfromtxt("data/test_labels.csv")

# Fit a model
depth = 5
clf = RandomForestClassifier(max_depth=depth)
clf.fit(X_train, y_train)

acc = clf.score(X_test, y_test)
print(acc)

metrics = """
Accuracy: {:10.4f}

![Confusion Matrix](plot.png)
""".format(acc)
with open("metrics.txt", "w") as outfile:
    outfile.write(metrics)

# Plot it
#disp = plot_confusion_matrix(confusion_matrix=clf, X_test, y_test)

#disp = ConfusionMatrixDisplay(confusion_matrix=cm,
#                              display_labels=display_labels)


# NOTE: Fill all variables here with default values of the plot_confusion_matrix
#disp = disp.plot(include_values=include_values,
#                 cmap=cmap, ax=ax, xticks_rotation=xticks_rotation)

#disp = ConfusionMatrixDisplay(confusion_matrix=clf)
#disp.plot(ax=ax, cmap=c)

#plt.savefig("plot.png")
