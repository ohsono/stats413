#!/bin/python
"""
====================================================
Project: SVM from scratch
Author : Hochan Son 
Date : 2025-12-14
Filename : SVM.py
How to run: 
     python prediction.py
====================================================
"""
import numpy as np

class SupportVectorMachine:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.C = lambda_param
        self.n_iters = n_iters
        self.w = 0
        self.b = 0

    def hingeloss(self, w, b, x, y):
        # Regularizer term
        reg = 0.5 * np.dot(w, w.T)

        loss = 0
        for i in range(x.shape[0]):
            # Optimization term
            opt_term = y[i] * ((np.dot(w, x[i])) + b)

            # calculating loss
            loss += max(0, 1 - opt_term)
            
        return (reg + self.C * loss).item()

    def fit(self, X, Y):
        # The number of features in X
        number_of_features = X.shape[1]

        # The number of Samples in X
        number_of_samples = X.shape[0]

        c = self.C
        learning_rate = self.learning_rate
        n_iters = self.n_iters

        # Creating ids from 0 to number_of_samples - 1
        ids = np.arange(number_of_samples)

        # Shuffling the samples randomly
        np.random.shuffle(ids)

        # creating an array of zeros
        w = np.zeros((1, number_of_features))
        b = 0
        losses = []

        # Gradient Descent logic
        for i in range(n_iters):
            # Calculating the Hinge Loss
            l = self.hingeloss(w, b, X, Y)

            # Appending all losses 
            losses.append(l)
            
            # Starting from 0 to the number of samples with batch_size as interval
            batch_size = 100 # Default batch size could be moved to __init__ or fit arg, kept local for now or use full batch? 
            # The original code had batch_size=100 in args. Let's keep it simple or default to 100 if not passed? 
            # The user plan said remove args. I'll hardcode or deduce. 
            # Original code was: def fit(self, X, Y, batch_size=100, learning_rate=0.001, epochs=1000):
            # Let's check the user request again. "Unify parameter usage". 
            # I will use self.n_iters for epochs. batch_size wasn't in __init__, so I'll leave it as a local constant or arg?
            # To be safe and clean, I'll keep batch_size as 100 here since it's not in init.
            
            for batch_initial in range(0, number_of_samples, batch_size):
                gradw = 0
                gradb = 0

                for j in range(batch_initial, batch_initial + batch_size):
                    if j < number_of_samples:
                        x = ids[j]
                        ti = Y[x] * (np.dot(w, X[x].T) + b)

                        if ti > 1:
                            gradw += 0
                            gradb += 0
                        else:
                            # Calculating the gradients

                            #w.r.t w 
                            gradw += c * Y[x] * X[x]
                            # w.r.t b
                            gradb += c * Y[x]

                # Updating weights and bias
                w = w - learning_rate * w + learning_rate * gradw
                b = b + learning_rate * gradb
        
        self.w = w
        self.b = b

        return self.w, self.b, losses

    def predict(self, X):
      prediction = np.dot(X, self.w[0]) + self.b # w.x + b
      return np.sign(prediction)