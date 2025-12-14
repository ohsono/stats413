#!/bin/python
"""
====================================================
Project: SVM from scratch
Author : Hochan Son 
Date : 2025-12-14
Filename : prediction.py
How to run: 
     python prediction.py
====================================================
"""
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from SVM import SupportVectorMachine


# Visualizing the scatter plot of the dataset
def visualize_dataset(x, y):
    plt.figure()
    plt.scatter(x[:, 0], x[:, 1], c=y)
    return plt.gcf()


# Visualizing SVM
def visualize_svm(x, x_test, y_test, w, b):

    def get_hyperplane_value(x, w, b, offset):
        return (-w[0][0] * x + b + offset) / w[0][1]

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.scatter(x_test[:, 0], x_test[:, 1], marker="o", c=y_test)

    x0_1 = np.amin(x_test[:, 0])
    x0_2 = np.amax(x_test[:, 0])

    x1_1 = get_hyperplane_value(x0_1, w, b, 0)
    x1_2 = get_hyperplane_value(x0_2, w, b, 0)

    x1_1_m = get_hyperplane_value(x0_1, w, b, -1)
    x1_2_m = get_hyperplane_value(x0_2, w, b, -1)

    x1_1_p = get_hyperplane_value(x0_1, w, b, 1)
    x1_2_p = get_hyperplane_value(x0_2, w, b, 1)

    ax.plot([x0_1, x0_2], [x1_1, x1_2], "y--")
    ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], "k")
    ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], "k")

    x1_min = np.amin(x)
    x1_max = np.amax(x)
    ax.set_ylim([x1_min - 3, x1_max + 3])

    return fig

def main():
  # Creating dataset
  X, y = datasets.make_blobs(
          n_samples = 100, # Number of samples
          n_features = 2, # Features
          centers = 2,
          cluster_std = 1,
          random_state=40
      )

  # Classes 1 and -1
  y = np.where(y == 0, -1, 1)

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

  svm = SupportVectorMachine(learning_rate=0.001, lambda_param=1.0, n_iters=1000)

  w, b, losses = svm.fit(X_train, y_train)

  prediction = svm.predict(X_test)

  # Loss value
  lss = losses.pop()

  print("Loss:", lss)
  print("Prediction:", prediction)
  print("Accuracy:", accuracy_score(prediction, y_test))
  print("w, b:", [w, b])


  # Visualizing the dataset and the SVM
  figure_1=visualize_dataset(X, y)
  figure_2=visualize_svm(X, X_test, y_test, w, b)

  figure_1.show()
  figure_2.show() 

  figure_1.savefig("STATS413_Final_p3_figure_1.png")
  figure_2.savefig("STATS413_Final_p3_figure_2.png")

if __name__ == "__main__":
  main()


  # Loss: 0.0991126738798482
  # Prediction: [-1.  1. -1. -1.  1.  1.  1.  1. -1.  1. -1. -1.  1.  1. -1. -1.  1. -1.
  #   1. -1.  1.  1. -1.  1.  1.  1. -1. -1. -1. -1.  1. -1. -1.  1.  1. -1.
  #   1. -1. -1.  1.  1.  1.  1.  1.  1. -1.  1.  1.  1.  1.]
  # Accuracy: 1.0
  # w, b: [array([[0.44477983, 0.15109913]]), 0.05700000000000004]
