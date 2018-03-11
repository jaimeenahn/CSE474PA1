import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt


def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer
    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer

    # Output:
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""

    return 1.0 / (1.0 + np.exp(-1.0 * z))


def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.
     Output:
     train_data: matrix of training set. Each row of train_data contains
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the
       training set
     test_data: matrix of training set. Each row of test_data contains
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set
     Things to do for preprocessing step:
     - remove features that have the same value for all data points
           with corresponding labels
     - convert original data set from integer to double by using double()
           function
     - normalize the data to [0, 1]
     - divide the original data set to training, validation and testing set"""
    n_valid = 5000
    #mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary
    mat = loadmat('mnist_sample.mat')

    length0 = len(mat['train0'])
    length1 = len(mat['train1'])
    length2 = len(mat['train2'])
    length3 = len(mat['train3'])
    length4 = len(mat['train4'])
    length5 = len(mat['train5'])
    length6 = len(mat['train6'])
    length7 = len(mat['train7'])
    length8 = len(mat['train8'])
    length9 = len(mat['train9'])



    train_data = np.concatenate((mat['train0'], mat['train1'],
                                 mat['train2'], mat['train3'],
                                 mat['train4'], mat['train5'],
                                 mat['train6'], mat['train7'],
                                 mat['train8'], mat['train9']), 0)
    train_label = np.concatenate((np.ones((mat['train0'].shape[0], 1), dtype='uint8'),
                                  2 * np.ones((mat['train1'].shape[0], 1), dtype='uint8'),
                                  3 * np.ones((mat['train2'].shape[0], 1), dtype='uint8'),
                                  4 * np.ones((mat['train3'].shape[0], 1), dtype='uint8'),
                                  5 * np.ones((mat['train4'].shape[0], 1), dtype='uint8'),
                                  6 * np.ones((mat['train5'].shape[0], 1), dtype='uint8'),
                                  7 * np.ones((mat['train6'].shape[0], 1), dtype='uint8'),
                                  8 * np.ones((mat['train7'].shape[0], 1), dtype='uint8'),
                                  9 * np.ones((mat['train8'].shape[0], 1), dtype='uint8'),
                                  10 * np.ones((mat['train9'].shape[0], 1), dtype='uint8')), 0)
    test_label = np.concatenate((np.ones((mat['test0'].shape[0], 1), dtype='uint8'),
                                 2 * np.ones((mat['test1'].shape[0], 1), dtype='uint8'),
                                 3 * np.ones((mat['test2'].shape[0], 1), dtype='uint8'),
                                 4 * np.ones((mat['test3'].shape[0], 1), dtype='uint8'),
                                 5 * np.ones((mat['test4'].shape[0], 1), dtype='uint8'),
                                 6 * np.ones((mat['test5'].shape[0], 1), dtype='uint8'),
                                 7 * np.ones((mat['test6'].shape[0], 1), dtype='uint8'),
                                 8 * np.ones((mat['test7'].shape[0], 1), dtype='uint8'),
                                 9 * np.ones((mat['test8'].shape[0], 1), dtype='uint8'),
                                 10 * np.ones((mat['test9'].shape[0], 1), dtype='uint8')), 0)
    test_data = np.concatenate((mat['test0'], mat['test1'],
                                mat['test2'], mat['test3'],
                                mat['test4'], mat['test5'],
                                mat['test6'], mat['test7'],
                                mat['test8'], mat['test9']), 0)

    # remove features that have same value for all points in the training data
    # convert data to double
    # normalize data to [0,1]

    # Split train_data and train_label into train_data, validation_data and train_label, validation_label
    # replace the next two lines
    # validation_data = np.array([])
    # validation_label = np.array([])

    validation_data = np.concatenate((mat['train0'][(length0-500):,:], mat['train1'][(length1-500):,:],
                                 mat['train2'][(length2-500):,:], mat['train3'][(length3-500):,:],
                                 mat['train4'][(length4-500):,:], mat['train5'][(length5-500):,:],
                                 mat['train6'][(length6-500):,:], mat['train7'][(length7-500):,:],
                                 mat['train8'][(length8-500):,:], mat['train9'][(length9-500):,:]), 0)
    validation_label = np.concatenate((np.ones((500,1), dtype='uint8'),
                                 2 * np.ones((500,1), dtype='uint8'),
                                 3 * np.ones((500,1), dtype='uint8'),
                                 4 * np.ones((500,1), dtype='uint8'),
                                 5 * np.ones((500,1), dtype='uint8'),
                                 6 * np.ones((500,1), dtype='uint8'),
                                 7 * np.ones((500,1), dtype='uint8'),
                                 8 * np.ones((500,1), dtype='uint8'),
                                 9 * np.ones((500,1), dtype='uint8'),
                                 10 * np.ones((500,1), dtype='uint8')), 0)

    train_data.astype(float)
    train_label.astype(float)
    test_data.astype(float)
    test_label.astype(float)
    validation_data.astype(float)
    validation_label.astype(float)
    print(test_data.dtype)


    print("preprocess done!")

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log
    %   likelihood error function with regularization) given the parameters
    %   of Neural Networks, thetraining data, their corresponding training
    %   labels and lambda - regularization hyper-parameter.
    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
    % Output:
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    # Your code here
    new_train = np.concatenate((training_data, np.ones((len(training_data), 1))), 1)
    n_train = len(training_data)  # number of input
    hidden = np.array(np.ones((n_train, n_hidden + 1)))
    output = np.zeros((n_train, n_class))
    label = np.zeros((n_train, n_class))

    for i in range(n_train):
        for mark in range(n_hidden):
            hidden[i][mark] = sigmoid(new_train[i].dot(np.transpose(w1[mark])))
        for marks in range(n_class):
            output[i][marks] = sigmoid(hidden[i].dot(np.transpose(w2[marks])))

    for i in range(n_train):
        label[i][training_label[i]-1] = 1
        for l in range(n_class):
            obj_val += label[i][l] * np.log(output[i][l]) + ((1 - label[i][l]) * np.log(1 - output[i][l]))

    obj_val = ((-1.0 * obj_val) / n_train)
    w1_val = 0
    w2_val = 0
    for j in range(n_hidden):
        for i in range(n_input + 1):
            w1_val += w1[j][i] * w1[j][i]
    for l in range(n_class):
        for j in range(n_hidden + 1):
            w2_val += w2[l][j] * w2[l][j]

    obj_val += lambdaval * (w1_val + w2_val) / (2 * n_train)

    grad_w1 = np.zeros((n_hidden, n_input + 1))
    grad_w2 = np.zeros((n_class, n_hidden + 1))
    new_hidden = np.empty_like(hidden)
    new_input = np.empty_like(new_train)
    new_w2 = np.empty_like(w2)
    new_hidden = np.delete(hidden, n_hidden, 1)
    new_w2 = np.delete(w2, n_hidden, 1)
    t_hidden = np.zeros((n_hidden, n_hidden))
    tt_hidden = np.zeros((n_hidden, 1))
    sumdeltaw2 = np.zeros((1, n_hidden))
    for i in range(n_train):
        sumdeltaw2 = np.dot((output[i].reshape(1, -1) - label[i].reshape(1, -1)), new_w2)
        grad_w2 += np.dot((output[i].reshape(1, -1) - label[i].reshape(1, -1)).T, hidden[i].reshape(1, -1))
        t_hidden = (np.ones((1, n_hidden)) - new_hidden[i].reshape(1, -1)) * new_hidden[i].reshape(1, -1)
        tt_hidden = t_hidden * sumdeltaw2
        grad_w1 += np.dot(tt_hidden.T, new_train[i].reshape(1, -1))

    grad_w1 = (grad_w1 + lambdaval * w1) / n_train
    grad_w2 = (grad_w2 + lambdaval * w2) / n_train

    #
    #
    #
    #

    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    #
    obj_grad = np.array([])
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()), 0)
    return (obj_val, obj_grad)


def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.
    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature
    %       vector of a particular image

    % Output:
    % label: a column vector of predicted labels"""

    labels = np.array([])
    # Your code here

    n_test = len(data)
    n_hidden_test = len(w1)
    n_class_test = len(w2)
    new_test = np.concatenate((data, np.ones((n_test, 1))), 1)
    hidden_test = np.array(np.ones((n_test, n_hidden_test + 1)))
    output_test = np.zeros((n_test, n_class_test))
    label_test = np.zeros((n_test, 1))

    for i in range(n_test):
        for mark in range(n_hidden_test):
            hidden_test[i][mark] = sigmoid(new_test[i].dot(np.transpose(w1[mark])))
        for marks in range(n_class_test):
            output_test[i][marks] = sigmoid(hidden_test[i].dot(np.transpose(w2[marks])))
            label_test[i][0] = np.argmax(output_test[i]) + 1

    labels = label_test
    labels.flatten()
    labels.T
    print(labels)

    return labels


"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 50

# set the number of nodes in output unit
n_class = 10

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden)
initial_w2 = initializeWeights(n_hidden, n_class)

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

# set the regularization hyper-parameter
lambdaval = 0

args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)
# Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter': 50}  # Preferred value.
nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

# In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
# and nnObjGradient. Check documentation for this function before you proceed.
# nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


# Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

# Test the computed parameters

predicted_label = nnPredict(w1, w2, train_data)
# find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, validation_data)

# find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, test_data)

# find the accuracy on Validation Dataset

print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')
