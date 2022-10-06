#-*-coding:utf-8-*- 
# author lyl

'''
使用手写K近邻实现手写数字识别
'''

from lylearn.knn import KNeighborsClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from loguru import logger
import random
import time


def main():
    # 手写数字集加载
    dataset = load_digits()
    data = dataset.data
    label = dataset.target
    target_names = dataset.target_names
    X_train, X_test, y_train, y_test = train_test_split(
        data, label, test_size=0.2
    )
    logger.info("训练集大小: {}, 测试集数据大小: {}".format(len(X_train), len(X_test)))
    model = KNeighborsClassifier(n_neighbors=5, p=2)
    model.fit(X_train, y_train)
    start_tm = time.time()
    test_score = model.score(X_test, y_test)
    span_time = time.time() - start_tm
    logger.info("test score: %.2f, use_time: %f" % (test_score, span_time))

    # predict
    index = [i for i in range(len(data))]
    random.shuffle(index)
    X = [data[index[0]]]
    y = label[index[0]]

    pred_num_name = target_names[model.predict(X)[0]]
    logger.info("predict number: {}, fact number: {}".format(pred_num_name, target_names[y]))


if __name__ == '__main__':
    main()