''' 문제 1 
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

x = np.array([1.0, 2.0, 3.0])
w = np.array([0.1, 0.2, 0.3])
b = 2.0

z = np.dot(w, x) + b     # 가중합 + 편향
a = sigmoid(z)           # 활성화

print(a)

#문제: 단일 층(Layer) 순전파 구현
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

x = np.array([1.0, 2.0, 3.0])
w=np.array  ([[0.1, 0.2, 0.3],[0.4, 0.5, 0.6]])
b = ([0.1,0.2])

z=np.dot(w,x)+b
a=sigmoid(z)

print(a)
'''
#문제: 2층 신경망 순전파 구현
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

x=np.array([1.0, 0.0])

w1=np.array([[0.1, 0.2],[0.3, 0.4]])
b1=np.array([0.1, 0.2])

w2=np.array([[0.5, 0.6]])
b2=np.array([0.3])

z1=np.dot(w1,x)+b1
h=sigmoid(z1)

z2=np.dot(w2,h)+b2
y=sigmoid(z2)

print(y)
