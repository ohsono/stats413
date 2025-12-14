#!/bin/python
"""
====================================================
Project: XGBoost from scratch
Author : Hochan Son 
Date : 2025-12-14
Filename : main.py
How to run: 
     python main.py
====================================================
"""
import numpy as np
import os
import time
import xgboost as xgb
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from xgboost_scratch import XGBoostModel, TreeBooster

class SquaredErrorObjective():
    def loss(self, y, pred): return np.mean((y - pred)**2)
    def gradient(self, y, pred): return pred - y
    def hessian(self, y, pred): return np.ones(len(y))

def main():

  X, y = fetch_california_housing(as_frame=True, return_X_y=True)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=43)

  # hyperparameters
  params = {
    'learning_rate': 0.1,
    'max_depth': 5,
    'subsample': 0.8,
    'reg_lambda': 1.5,
    'gamma': 0.0,
    'min_child_weight': 25,
    'base_score': 0.0,
    'tree_method': 'exact',
  }
  num_boost_round = 50

  # train the from-scratch XGBoost model
  start_time = time.time()
  model_scratch = XGBoostModel(params, random_seed=42)
  model_scratch.fit(X_train, y_train, SquaredErrorObjective(), num_boost_round)
  scratch_train_time = time.time() - start_time

  # train the library XGBoost model
  start_time = time.time()
  dtrain = xgb.DMatrix(X_train, label=y_train)
  dtest = xgb.DMatrix(X_test, label=y_test)
  model_xgb = xgb.train(params, dtrain, num_boost_round)
  xgb_train_time = time.time() - start_time

  # predict 
  pred_scratch = model_scratch.predict(X_test)
  pred_xgb = model_xgb.predict(dtest)
  print(f'scratch score: {SquaredErrorObjective().loss(y_test, pred_scratch)}')
  print(f'xgboost score: {SquaredErrorObjective().loss(y_test, pred_xgb)}')
  print(f'\nscratch training time: {scratch_train_time:.2f}s')
  print(f'xgboost training time: {xgb_train_time:.2f}s')
  print(f'speedup: {scratch_train_time/xgb_train_time:.2f}x')
if __name__ == "__main__":
  main()

# scratch score: 0.2434125759558149
# xgboost score: 0.24385224475634615

# scratch training time: 29.99s
# xgboost training time: 0.18s
# speedup: 166.18x
