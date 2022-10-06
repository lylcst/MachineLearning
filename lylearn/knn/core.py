#-*-coding:utf-8-*- 
# author lyl
import numpy as np
from collections import Counter
from typing import Union, List


def L_distance(x, y, p=2):
    assert len(x) == len(y)
    if len(x) < 1:
        return 0
    return np.power(np.sum(np.power(abs(np.array(x) - np.array(y)), p)), 1 / p)


class KNode:
    def __init__(self, instance, split_dim, label, left=None, right=None):
        self.instance = instance
        self.split_dim = split_dim
        self.label = label
        self.left = left
        self.right = right

def create_kdtree(instance_list: List, split_dim=0) -> Union[KNode, None]:
    '''
    :param instance_list: [([3,2,3], 0), ([4,3,5], 1)]
    :param split_dim: 0
    :return: KNode
    '''
    if len(instance_list) == 0:
        return None
    instance_list = sorted(instance_list, key=lambda x:x[0][split_dim])
    split_index = len(instance_list) // 2
    next_split_dim = (split_dim+1)%len(instance_list[0][0])
    return KNode(
        instance=instance_list[split_index][0],
        split_dim=split_dim,
        label=instance_list[split_index][1],
        left=create_kdtree(instance_list[:split_index], split_dim=next_split_dim),
        right=create_kdtree(instance_list[split_index+1:], split_dim=next_split_dim)
    )


def get_topk_instance(root, ins, k=1):
    '''
    :param root:
    :param k:
    :return: [], []
    '''
    if root is None:
        return []

    def update_max_dist(instance_list, ins):
        max_index = 0
        max_val = float('-inf')
        for index, instance in enumerate(instance_list):
            dist = L_distance(ins, instance, p=2)
            if dist > max_val:
                max_index = index
                max_val = dist
        return max_val, max_index

    def process(root, ins, k, info) -> (list, list, int, float):
        if root is None:
            return info
        split_dim = root.split_dim
        instance = root.instance
        label = root.label
        if ins[split_dim] <= instance[split_dim]:
            first_child = root.left
            second_child = root.right
        else:
            first_child = root.right
            second_child = root.left
        instance_list, label_list, max_index, max_dist = process(first_child, ins, k, info)
        if len(instance_list) < k:
            instance_list.append(instance)
            label_list.append(label)
            dist = L_distance(ins, instance, p=2)
            if dist < max_dist:
                max_dist, max_index = update_max_dist(instance_list, ins)
        else:
            dist = L_distance(ins, instance, p=2)
            if dist < max_dist:
                instance_list[max_index] = instance
                label_list[max_index] = label
                max_dist, max_index = update_max_dist(instance_list, ins)
        info = (instance_list, label_list, max_index, max_dist)
        if abs(ins[split_dim] - instance[split_dim]) <= max_dist:
            return process(second_child, ins, k, info)
        return info
    init_info = ([], [], -1, float('inf'))
    result =  process(root, ins, k, init_info)
    return result[0], result[1]


def preorder_kbtree(root):
    if root is not None:
        print(root.instance, root.label)
        preorder_kbtree(root.left)
        preorder_kbtree(root.right)


class KNeighborsClassifier:
    def __init__(self, n_neighbors=5, p=2):
        self.n_neighbors = n_neighbors
        self.p = p

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.kd_tree = create_kdtree(list(zip(X_train, y_train)), split_dim=0)

    def pure_predict(self, X_list: Union[List[List], List]) -> List:
        result = []
        for X in X_list:
            knn_list = []
            for i in range(self.n_neighbors):
                knn_list.append((L_distance(X, self.X_train[i], self.p), self.y_train[i]))

            for i in range(self.n_neighbors, len(self.X_train)):
                distance = L_distance(X, self.X_train[i])
                max_index = knn_list.index(max(knn_list, key=lambda x: x[0]))
                if distance < knn_list[max_index][0]:
                    knn_list[max_index] = (distance, self.y_train[i])
            y_preds = [item[-1] for item in knn_list]
            counter = Counter(y_preds)
            result.append(sorted(counter.items(), key=lambda x: x[1])[-1][0])
        return result

    def predict(self, X_list):
        result = []
        for X in X_list:
            _, label_list = get_topk_instance(self.kd_tree, X, self.n_neighbors)
            result.append(
                sorted(Counter(label_list).items(), key=lambda x: x[1])[-1][0]
            )
        return result

    def score(self, X_test, y_test):
        pred = self.predict(X_test)
        return np.sum(np.array(pred)==np.array(y_test))/np.size(y_test)



