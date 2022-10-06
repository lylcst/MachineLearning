#-*-coding:utf-8-*- 
# author lyl
import numpy as np
from functools import reduce
from collections import Counter

class GaussianNB:
    def __init__(self, eps=1e-5):
        self.eps = eps

    def fit(self, X, y):
        assert len(X) == len(y)
        self.classes_set = list(set(y))
        classes_dict = {}

        for idx, data in enumerate(X):
            classes_dict[y[idx]] = classes_dict.get(y[idx], [])
            classes_dict[y[idx]].append(data)
        assert len(self.classes_set) == len(classes_dict)
        # 计算均值和方差
        for class_id, data in classes_dict.items():
            x_mean = np.mean(data, axis=0, keepdims=False)
            x_std = np.sum((data-x_mean)**2, axis=0) / len(data)
            classes_dict[class_id] = {
                "data": data,
                "mean": x_mean,
                "std": x_std,
                "prob": len(data)/len(X)
            }
        self.class_dict = classes_dict

    def gaussian_distribution_prob(self, x, mean, std):
        expon = -(x-mean)**2 / (2*std + self.eps)
        return 1/np.sqrt(2*np.pi*std+self.eps) * np.exp(expon)

    def predict(self, X_list):
        result = []
        for X in X_list:
            tmp = []
            for class_id in self.classes_set:
                multi_prob = self.gaussian_distribution_prob(X,
                                                self.class_dict[class_id]['mean'],
                                                self.class_dict[class_id]['std'])
                prob = reduce(lambda a, b: a*b, multi_prob)
                tmp.append(self.class_dict[class_id]["prob"]*prob)
            result.append(self.classes_set[np.argmax(tmp).item()])
        return result

    def score(self, X_list, y_list):
        X_pred = self.predict(X_list)
        return np.sum(X_pred == y_list) / len(y_list)


class MultinomialNB:
    def __init__(self, lambd=1):
        self.lambd = lambd

    def fit(self, X, y):
        assert len(X) == len(y)
        self.classes_set = list(set(y))
        classes_dict = {}
        for idx, data in enumerate(X):
            classes_dict[y[idx]] = classes_dict.get(y[idx], [])
            classes_dict[y[idx]].append(data)
        for class_id, data in classes_dict.items():
            data = np.array(data)
            jl_list = []
            for val in range(len(data[0])):
                jl_list.append(dict(Counter(data[:,val])))
            classes_dict[class_id] = {
                "data": data,
                "prob": len(data)/len(X),
                "jl_list": jl_list
            }
        self.classes_dict = classes_dict

    def predict(self, X_list):
        result = []
        for X in X_list:
            tmp = []
            for class_id in self.classes_set:
                attr_prob = 1
                data_len = len(self.classes_dict[class_id]["data"])
                for idx, item in enumerate(self.classes_dict[class_id]["jl_list"]):
                    attr_prob *= (item.get(X[idx], 0) + self.lambd) / (data_len + len(item)*self.lambd)
                prob = self.classes_dict[class_id]["prob"] * attr_prob
                tmp.append(prob)
            result.append(self.classes_set[np.argmax(tmp).item()])
        return result

    def score(self, X_list, y_list):
        y_pred = self.predict(X_list)
        count = np.sum([i == k for i, k in zip(y_pred, y_list)])
        return count / len(y_list)
