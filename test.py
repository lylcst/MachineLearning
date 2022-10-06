#-*-coding:utf-8-*- 
# author lyl
# from lylearn.knn import create_kdtree, preorder_kbtree, get_topk_instance
#
#
# instance_list = [[2, 3],
#                  [1, 2],
#                  [3, 2],
#                  [8, 2],
#                  [5, 2]]
#
# label = [1, 0, 2, 3, 4]
#
# root = create_kdtree(list(zip(instance_list, label)), split_dim=0)
# preorder_kbtree(root)
#
# print(get_topk_instance(root, [5, 4], k=1))

#import numpy as np
# from collections import Counter
#
# a = [
#     [2,4,1,5],
#     [4,5,7,3],
#     [7,5,3,0],
#     [5,4,3,2],
#     [5,4,4,3]
# ]
#
# x_mean = np.mean(a, axis=0, keepdims=True)
#
# print(np.sum((a - x_mean)**2, axis=0))
#
# print(Counter(np.array(a)[:, 0])[5])


'''
x^7 + 3*x^5 + 5*x^3 + 7x = 2022

'''

#
# def process(val=0, thresh=1e-5):
#     def calc(x):
#         return pow(x, 7) + 3*pow(x,5) + 5*pow(x, 3) + 7*x - 2022
#     def daoshu(x):
#         return 7*pow(x, 6) + 15*pow(x,4) + 15*pow(x, 2) + 7
#     while abs(calc(val) - 0) > thresh:
#         val = val - calc(val)/daoshu(val)
#     return val
#
#
# x = process(0)
# print(x)
# res = pow(x, 7) + 3*pow(x,5) + 5*pow(x, 3) + 7*x
# print(res)
#

from graphviz import Digraph


g = Digraph('G', filename='text.gv')

g.node('node1', label='你好', fontname="FangSong")
g.node('node2', label='world')
g.edge('node1', 'node2')

g.view()



from torch.nn import CrossEntropyLoss




