#-*-coding:utf-8-*- 
# author lyl
from lylearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


# 高斯朴素贝叶斯
X, y = load_digits(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

gnb = GaussianNB()
gnb.fit(X_train, y_train)
print("分类准确率：%.2f" % gnb.score(X_test, y_test))


# 多项式朴素贝叶斯
mnb = MultinomialNB()
X = [
    [1, 'S'],
    [1, 'M'],
    [1, 'M'],
    [1, 'S'],
    [1, 'S'],
    [2, 'S'],
    [2, 'M'],
    [2, 'M'],
    [2, 'L'],
    [2, 'L'],
    [3, 'L'],
    [3, 'M'],
    [3, 'M'],
    [3, 'L'],
    [3, 'L']
]
y = [-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1]

mnb.fit(X, y)
pred = mnb.predict([[2, 'S'], [3, 'L']])
print(pred)
