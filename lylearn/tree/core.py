#-*-coding:utf-8-*- 
# author lyl
import numpy as np
from collections import Counter
from graphviz import Digraph


class TreeNode:
    def __init__(self, attr=None, child=None, isleaf=True):
        self.attr = attr
        self.child = child
        self.isleaf = isleaf


class DecisionTreeClassifier:
    def __init__(self, attr=None):
        self.attr = attr
        self.flag = True if attr is not None else False
        self.thresh = 0.01
        self.root = None

        # 可视化决策树
        self.leaf_count = 0
        self.node_count = 0

    def entropy(self, y):
        counter = Counter(y)
        return np.sum([-k/len(y)*np.log2(k/len(y)) for
                       k in counter.values()])

    def max_category_y(self, y):
        return sorted(Counter(y).items(), key=lambda x:x[1])[-1][0]

    def create_tree(self, X, y, attrs):
        assert len(X) == len(y) and len(y) > 0
        counter = Counter(y)
        if len(counter) == 1:
            return TreeNode(attr=y[0])
        # 如果属性都使用了返回
        if len(attrs) == len(self.attr):
            return TreeNode(attr=self.max_category_y(y), isleaf=True)

        H_D = self.entropy(y)
        # 计算不同特征的条件熵
        A_entropy_dict = []
        for k in range(len(X[0])):
            A_dict = {}
            for j in range(len(X)):
                if X[j][k] not in A_dict:
                    A_dict[X[j][k]] = [[X[j]], [y[j]]]
                else:
                    A_dict[X[j][k]][0].append(X[j])
                    A_dict[X[j][k]][1].append(y[j])
            A_entropy = 0
            for key, val in A_dict.items():
                A_entropy = A_entropy + self.entropy(val[1]) * len(val[1]) / len(y)
            A_entropy_dict.append({
                'A_dict': A_dict,
                'A_entropy': A_entropy
            })
        # 计算信息增益
        A_entropy_zy = [H_D-item['A_entropy'] for item in A_entropy_dict]
        index = np.argmax(A_entropy_zy).item()
        if A_entropy_zy[index] < self.thresh:
            return TreeNode(attr=self.max_category_y(A_entropy_dict[index]['A_dict'][1]))

        root = TreeNode(attr=self.attr[index], child=dict(), isleaf=False)
        for k, v in A_entropy_dict[index]['A_dict'].items():
            root.child[k] = self.create_tree(v[0], v[1], attrs=attrs+[self.attr[index]])
        return root

    def plot_tree(self):
        graph = Digraph('decision_tree', filename='decisionTree.dot')
        def process(tree):
            if tree.isleaf:
                name = f'leaf_{self.leaf_count}'
                self.leaf_count += 1
                graph.node(name, label=tree.attr, fontname="FangSong", shape='record')
                return name
            name = f'node_{self.node_count}'
            self.node_count += 1
            graph.node(name, label=tree.attr, fontname="FangSong")
            for edge_label, child in tree.child.items():
                graph.edge(name,
                           process(child),
                           label=edge_label,
                           fontname="FangSong")
            return name
        process(self.root)
        self.leaf_count = 0
        self.node_count = 0
        graph.view()

    def fit(self, X, y):
        assert len(X) == len(y) and len(y) > 0
        if self.attr is not None:
            assert len(X[0]) == len(self.attr)
        else:
            self.attr = list(range(len(X[0])))
        self.root = self.create_tree(X, y, attrs=[])

    def traverse(self, root, features):
        if root.isleaf:
            return root.attr
        attr = features[root.attr] if not self.flag else features[self.attr.index(root.attr)]
        return self.traverse(root.child[attr], features)

    def predict(self, X_list):
        assert self.root is not None, 'decision tree is None'
        result_list = []
        for X in X_list:
            root = self.root
            result = self.traverse(root, X)
            result_list.append(result)
        return result_list

    def score(self, X, y):
        y_pred = self.predict(X)
        count = np.sum([i == k for i, k in zip(y_pred, y)])
        return count / len(y)


