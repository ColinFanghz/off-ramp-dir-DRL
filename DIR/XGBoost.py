"""
@Author: Fhz
@Create Date: 2023/7/13 15:37
@File: XGBoost.py
@Description:
@Modify Person Date:
"""
import torch
import random
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix
from sklearn import preprocessing
import pickle
import datetime
from time import time
import matplotlib.pyplot as plt



if __name__ == '__main__':

        y_train_path = "../data_process/y_train.npy"
        x_train_path = "../data_process/X_train.npy"
        y_test_path = "../data_process/y_test.npy"
        x_test_path = "../data_process/X_test.npy"
        
        X_train = np.load(file=x_train_path)
        y_train = np.load(file=y_train_path)
        
        X_test = np.load(file=x_test_path)
        y_test = np.load(file=y_test_path)
                        
        X_train = X_train.reshape((len(X_train), -1))
        X_test = X_test.reshape((len(X_test), -1))
        print(X_test.shape)
        
        
        XGB = XGBClassifier(n_estimators=150, learning_rate=0.15, max_depth=8, num_class=3, objective='multi:softmax', gpu_id=1)
        XGB.fit(X_train, y_train)
        
        model_name = "xgboost.pickle"
        print(model_name)
        pickle.dump(bst, open(model_name, "wb"))
            
            
        preds = bst.predict(X_test)
        print("accuracy_score:")
        print(accuracy_score(y_test, preds))
        
        print("==========================")
        
        
        print("precision_score 0:")
        print(precision_score(preds, y_test, labels=[0], average="macro"))
        print("precision_score 1:")
        print(precision_score(preds, y_test, labels=[1], average="macro"))
        print("precision_score 2:")
        print(precision_score(preds, y_test, labels=[2], average="macro"))
        
        print("==========================")
        
        print("recall_score 0:")
        print(recall_score(preds, y_test, labels=[0], average="macro"))
        print("recall_score 1:")
        print(recall_score(preds, y_test, labels=[1], average="macro"))
        print("recall_score 2:")
        print(recall_score(preds, y_test, labels=[2], average="macro"))
        
        print("==========================")
        
        print("f1_score 0:")
        print(f1_score(preds, y_test, labels=[0], average="macro"))
        print("f1_score 1:")
        print(f1_score(preds, y_test, labels=[1], average="macro"))
        print("f1_score 2:")
        print(f1_score(preds, y_test, labels=[2], average="macro"))
        
        print("==========================")
        
        print(confusion_matrix(y_test, preds, labels=[0, 1, 2]))
               
        print(datetime.datetime.fromtimestamp(time() - time0).strftime("%M:%S:%f"))
                    
          