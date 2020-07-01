import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

np.random.seed(4)
data = pd.read_csv("data_classification.csv", header = None)

true_x = []
true_y = []

false_x = []
false_y = []

for element in data.values:
    if element[2] == 1:
        true_x.append(element[0])
        true_y.append(element[1])
    else:
        false_x.append(element[0])
        false_y.append(element[1])

plt.scatter(true_x, true_y, c="b", marker="o")
plt.scatter(false_x, false_y, c = "r", marker="x")
# plt.show()

def sigmoid(z):
    return 1.0 / ( 1 + np.exp(-z))

def predict(proba):
    if proba > 0.5:
        return 1
    else:
        return 0

def predict_proba(feature, weight):
    z = np.dot(feature, weight)
    return sigmoid(z)

def cost_function(feature, weight, label):
    """
    :param feature: (100 * 3)
    :param weight:  (3 * 1)
    :param label:   (100 * 1)
    :return:        function cost
    """
    n = len(label)
    prediction = predict_proba(feature, weight)

    cost_class1 = -label * np.log(prediction)
    cost_class0 = -(1 - label) * np.log(1 - prediction)

    cost = cost_class1 + cost_class0

    return cost.sum()/n

def update_weight(feature , weight, label, learning_rate):
    """
    :param feature:         (100 * 3)
    :param weight:          (3 * 1)
    :param label:           (100 * 1)
    :param learning_rate:   float learning rate=
    :return:                float new weight =
    """
    n = len(label)
    prediction = predict_proba(feature, weight)
    label = label.reshape((len(label), 1))
    gradient = np.dot(feature.T, (prediction - label))
    gradient = gradient/ n
    update_weight = weight - gradient * learning_rate
    return update_weight

def training(feature, weight, label, learning_rate, iteration):
    cost_history = []
    for i in range(iteration):
        weight = update_weight(feature , weight, label, learning_rate)
        # print("==================",weight)
        cost = cost_function(feature, weight, label)
        cost_history.append(cost)
    return weight, cost_history


def logistic_sigmoid_regression(X_train, y_train):
    weights, cost_history = training(feature=X_train, label=y_train, weight=np.random.rand(3,1), learning_rate=0.001, iteration=30)
    return weights, cost_history

if __name__ == "__main__":
    D_feature = []
    D_label = []
    for ele in data.values:
        D_feature.append(ele[:2])
        D_label.append(ele[-1])
    D_feature = np.asarray(D_feature)
    matrix_ones = np.ones((D_feature.shape[0],1))
    D_new_feature = np.hstack((matrix_ones, D_feature))
    D_label = np.asarray(D_label)
    X_train, X_test, y_train, y_test = train_test_split(D_new_feature, D_label, test_size=0.3, random_state=42)
    weight, cost_hs = logistic_sigmoid_regression(D_new_feature, D_label)
    predict_probability = predict_proba(X_test, weight)
    y_predict = predict(predict_probability)