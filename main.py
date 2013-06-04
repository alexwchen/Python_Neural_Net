from nn import *
from scipy.io import loadmat
from scipy.io import savemat
from sklearn import cross_validation
import random

data = loadmat('./data_12_gesture_v1.mat')
X = data['X']
y = data['y']

for idx in range(1000):

    print '================================================================'

    lamb = idx*0.01
    print 'Lambda: ' + str(lamb)

    # It turned out that 300 nodes for hidden feature is the best combination
    T1 = initialize_weights(400, 100)
    T2 = initialize_weights(100, 12)
    print 'T1: ', T1.shape
    print 'T2: ', T2.shape

    alpha = 0.1
    for i in range(300):
        Theta1_grad, Theta2_grad , J= backwardpropagation(T1, T2, X, y, lamb)
        T1 = T1 - alpha*Theta1_grad
        T2 = T2 - alpha*Theta2_grad
        print 'Iteration: ' + str(i) + ' Cost: ' + str(J)


    prediction, full_matrix = predict(T1, T2, X)
    print 'Accuracy: ', sum((prediction) == y)/float(y.shape[0])

    data['Theta1'] = T1
    data['Theta2'] = T2
    savemat('para_12_gesture_v1.mat', data)
    break
