"""
@Author: Fhz
@Create Date: 2024/1/10 12:22
@File: ensemble.py
@Description:
@Modify Person Date:
"""
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier
from time import time
import datetime
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix
import pickle
import logging


def log():
    # 创建一个logger对象
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.DEBUG)

    # 创建一个文件处理器
    handler = logging.FileHandler('my_log_RF.txt')
    handler.setLevel(logging.DEBUG)

    # 创建一个格式化器
    # formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter()
    handler.setFormatter(formatter)

    # 将处理器添加到logger对象中
    logger.addHandler(handler)

    return logger


if __name__ == '__main__':

    logger = log()

    models_name = ["RF.pickle"]

    y_train_path = "../data_process/y_train.npy"
    x_train_path = "../data_process/X_train.npy"
    y_test_path = "../data_process/y_test.npy"
    x_test_path = "../data_process/X_test.npy"

    X_train = np.load(file=x_train_path)
    y_train = np.load(file=y_train_path)

    X_test = np.load(file=x_test_path)
    y_test = np.load(file=y_test_path)

    # X_train = X_train.reshape((len(X_train), -1))
    # X_test = X_test.reshape((len(X_test), -1))
    print(X_test.shape)

    for ii in range(len(models_name)):
        X_train1 = X_train[:, :50, :]
        X_test1 = X_test[:, :50, :]

        X_train1 = X_train1.reshape((len(X_train1), -1))
        X_test1 = X_test1.reshape((len(X_test1), -1))


        time0 = time()
        logger.info("Starting Training!")

        RF = RandomForestClassifier(n_estimators=150, max_depth=8)
        RF.fit(X_train1, y_train)
        # RF = pickle.load(open("RF.pickle", "rb"))


        model_name = models_name[ii]
        logger.info(model_name)
        pickle.dump(RF, open(model_name, "wb"))


        preds = RF.predict(X_test1)
        logger.info("accuracy_score:")
        logger.info(accuracy_score(y_test, preds))

        logger.info("==========================")

        logger.info("precision_score 0:")
        logger.info(precision_score(preds, y_test, labels=[0], average="macro"))
        logger.info("precision_score 1:")
        logger.info(precision_score(preds, y_test, labels=[1], average="macro"))
        logger.info("precision_score 2:")
        logger.info(precision_score(preds, y_test, labels=[2], average="macro"))

        logger.info("==========================")

        logger.info("recall_score 0:")
        logger.info(recall_score(preds, y_test, labels=[0], average="macro"))
        logger.info("recall_score 1:")
        logger.info(recall_score(preds, y_test, labels=[1], average="macro"))
        logger.info("recall_score 2:")
        logger.info(recall_score(preds, y_test, labels=[2], average="macro"))

        logger.info("==========================")

        logger.info("f1_score 0:")
        logger.info(f1_score(preds, y_test, labels=[0], average="macro"))
        logger.info("f1_score 1:")
        logger.info(f1_score(preds, y_test, labels=[1], average="macro"))
        logger.info("f1_score 2:")
        logger.info(f1_score(preds, y_test, labels=[2], average="macro"))

        logger.info("==========================")

        # print(confusion_matrix(y_test, preds, labels=[0, 1, 2]))

        logger.info(datetime.datetime.fromtimestamp(time() - time0).strftime("%M:%S:%f"))
        logger.info("==========================")
        logger.info("==========================")
        logger.info("==========================")
        logger.info("==========================")

