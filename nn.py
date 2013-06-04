from numpy import *
from scipy import *
from datetime import datetime

def sigmoid(z):
    g = 1.0 / (1.0 + exp(-z))
    return g

def sigmoid_gradient(z):
    g = sigmoid(z)*(1-sigmoid(z))
    return g


def initialize_weights(L_in, L_out):
    #random.seed(datetime.now()[-1])
    epsilon_init = 0.12
    W = rand(L_out, 1+L_in)*2*epsilon_init - epsilon_init
    return W

def debug_init_weights(L_in, L_out):
    L_out = L_out + 1
    w = zeros((L_in*L_out, 1))
    w[:,0] = sin(arange(L_in*L_out)+1)
    w = w.reshape((L_out, L_in)) / 10
    return transpose(w)

def compute_numercial_gradient(T1, T2, X, y, lamb):
    T1_ = T1.transpose()
    T2_ = T2.transpose()

    # Reconstruct T1, T2 into a vector
    theta1 = T1_.reshape((1, T1.shape[0]*T1.shape[1]))
    theta2 = T2_.reshape((1, T2.shape[0]*T2.shape[1]))
    theta = hstack((theta1, theta2))

    numgrad = zeros((theta.shape[0], theta.shape[1]))
    perturb = zeros((theta.shape[0], theta.shape[1]))
    e = 1e-4

    for p  in range(theta.shape[1]):
        # Set perturbation vector
        perturb[0,p] = e
        new_theta = theta - perturb

        new_theta1 = zeros((1,(T1.shape[0]*T1.shape[1])))
        new_theta1[0,:] = new_theta[0,:(T1.shape[0]*T1.shape[1])]
        new_theta1 = new_theta1.transpose()
        new_theta1 = new_theta1.reshape((T1.shape[0], T1.shape[1]))

        new_theta2 = zeros((1,(T2.shape[0]*T2.shape[1])))
        new_theta2[0,:] = new_theta[0,(T1.shape[0]*T1.shape[1]):]
        new_theta2 = new_theta2.transpose()
        new_theta2 = new_theta2.reshape((T2.shape[0], T2.shape[1]))

        loss1 = nn_cost_function(new_theta1, new_theta2, X, y, lamb)

        new_theta = theta + perturb

        new_theta1 = zeros((1,(T1.shape[0]*T1.shape[1])))
        new_theta1[0,:] = new_theta[0,:(T1.shape[0]*T1.shape[1])]
        new_theta1 = new_theta1.reshape((T1.shape[0], T1.shape[1]))

        new_theta2 = zeros((1,(T2.shape[0]*T2.shape[1])))
        new_theta2[0,:] = new_theta[0,(T1.shape[0]*T1.shape[1]):]
        new_theta2 = new_theta2.reshape((T2.shape[0], T2.shape[1]))

        loss2 = nn_cost_function(new_theta1, new_theta2, X, y, lamb)

        # Compute Numerical Gradient
        numgrad[0,p] = (loss2 - loss1) / (2*e)
        perturb[0,p] = 0
        #if p==1:
        #    break
    return numgrad


def predict(Theta1, Theta2, X):
    vector_ones = ones(X.shape[0]).reshape(X.shape[0],1)

    # input layer
    a1 = hstack((vector_ones,X))

    # middle layer
    Z1 = dot(a1, Theta1.transpose())
    a2 = sigmoid(Z1)
    a2 = hstack((vector_ones,a2))

    # final layer
    Z2 = dot(a2, Theta2.transpose())
    a3 = sigmoid(Z2)

    agmax = (a3.argmax(axis=1)+1).reshape(X.shape[0],1)
    return agmax, a3

def nn_cost_function(Theta1, Theta2, X, y, lamb):
    predictions, full_matrix = predict(Theta1, Theta2, X)

    # first half of the equation ( line 1)
    J = zeros((full_matrix.shape[0],1))
    for i in range(full_matrix.shape[1]):
        idx = i+1
        new_y = (y==idx)
        log1 = log(full_matrix[:,i]).reshape(full_matrix.shape[0],1)
        log2 = log(1-full_matrix[:,i]).reshape(full_matrix.shape[0],1)
        s1 = -(new_y*log1)
        s2 = -(1-new_y)*log2
        J = J + s1 + s2
    J = sum(J)/full_matrix.shape[0]

    # 2nd half of the equation ( line 1)
    Reg1 = sum(pow(Theta1[:, 1:],2))
    Reg2 = sum(pow(Theta2[:, 1:],2))
    reg_total =  (Reg1+Reg2)*(lamb/float(2)/full_matrix.shape[0])
    J = J + reg_total
    return J

def backwardpropagation(Theta1, Theta2, X, y, lamb):
    Theta1_grad = zeros((Theta1.shape[0], Theta1.shape[1]))
    Theta2_grad = zeros((Theta2.shape[0], Theta2.shape[1]))
    m = X.shape[0]

    ####################################################
    # COST FUNCTION
    ####################################################
    J = nn_cost_function(Theta1, Theta2, X, y,lamb)
    #print J

    ####################################################
    # FEED FOWARD PROPAGATION
    ####################################################
    vector_ones = ones(m).reshape(X.shape[0],1)

    # input layer
    a1 = hstack((vector_ones,X))

    # middle layer
    Z1 = dot(a1, Theta1.transpose())
    a2 = sigmoid(Z1)
    a2 = hstack((vector_ones,a2))

    # final layer
    Z2 = dot(a2, Theta2.transpose())
    a3 = sigmoid(Z2)

    ####################################################
    # BACKARD PROPAGATION
    ####################################################

    # Calculate delta3
    delta3 = zeros((m, a3.shape[1]))
    for j in range(a3.shape[1]):
        new_y = (y==j+1)
        # these reshpaes are tricky numpy dimension hacks switching between (m,) & (m,1)
        delta3[:,j] = (a3[:,j].reshape(m,1) - new_y).reshape(m,)

    # Calculate delta2
    delta2 = dot(delta3,Theta2)
    delta2 = delta2[:, 1:]
    diffZ = sigmoid_gradient(Z1)
    delta2 = delta2*diffZ

    # Calculate Delta (Accumulative Gradient)
    Theta2_grad = Theta2_grad + dot(delta3.transpose(),a2)
    Theta1_grad = Theta1_grad + dot(delta2.transpose(),a1)


    Theta2_grad = Theta2_grad/m
    Theta1_grad = Theta1_grad/m

    Theta2_grad[:, 1:] = Theta2_grad[:, 1:] + Theta2[:, 1:]*(lamb/m)
    Theta1_grad[:, 1:] = Theta1_grad[:, 1:] + Theta1[:, 1:]*(lamb/m)

    return Theta1_grad, Theta2_grad, J


