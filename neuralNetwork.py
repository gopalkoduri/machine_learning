#!/usr/bin/env python

#author: Gopala Krishna Koduri
#mail: gopala.koduri@gmail.com

from __future__ import division
from scipy.optimize import fmin_bfgs, fmin_cg, fmin_ncg
import numpy as np

count = 0

def sigmoid(x):
    """
    returns 1/(1+e^(-x)), this function works with matrices and np.arrays by
    default.
    """
    return 1/(1+np.exp(-x))

def sigmoidGradient(x):
    """
    this function returns sigmoid(x).*(1-sigmoid(x)), to be used in back
    propgation procedure.
    """
    tmp = sigmoid(x)
    return np.multiply(tmp, 1-tmp)

def randInitializeWeights(prevLayerSize, curLayerSize):
    eInit = 0.12
    weights = np.random.rand(curLayerSize*(prevLayerSize+1))*2*eInit - eInit
    return weights

def cost(nnParams, hiddenLayerSizes,\
         numLabels, X, y, _lambda):
    """
    nnParams: Unrolled theta/weights.
    hiddenLayerSizes: Number of units in each hidden layer. It should be an
    np.array/list.
    numLabels: Number of class labels, which is same as number of units in
    output layer.
    X: feature np.matrix, must NOT include np.ones as first column.
    y: column np.matrix of labels, each of them should be an integer between 0 and
    numLabels.
    _lambda: regularization parameter.

    """
    global count
    count = count+1
    inputLayerSize = X.shape[1]
    numHiddenLayers = len(hiddenLayerSizes)

    #roll nnParams into thetas based on layer sizes 
    allTheta = [] #np.array of theta matrices

    layerSizes = []
    layerSizes.extend(hiddenLayerSizes)
    layerSizes.append(numLabels)

    prevLayerSize = inputLayerSize
    startIndex = 0
    for layerNum in xrange(numHiddenLayers+1):
        curLayerSize = layerSizes[layerNum]
        numValues = curLayerSize * (prevLayerSize+1)
        endIndex = startIndex+numValues
        theta = nnParams[startIndex:endIndex]
        theta = theta.reshape(curLayerSize, prevLayerSize+1, order="F")
        theta = np.matrix(theta)
        allTheta.append(theta)
        startIndex=endIndex
        prevLayerSize = curLayerSize

    numSamples = X.shape[0]
    J = 0 #is cost
    predictions = np.zeros(numSamples)
    

    #compute activations using thetas and inputs
    allActivations = []
    allZs = []
    #use numpy.np.insert to add column of np.ones
    #np.insert(np.matrix/arr, indexToInsertBefore, axis=0(row)/1(column))
    layerInput = np.insert(X, 0, np.ones(numSamples), axis=1)
    allActivations.append(X) #first layer activations are inputs

    for layerNum in xrange(numHiddenLayers+1):
        theta = allTheta[layerNum]
        z = layerInput*theta.T
        activation = sigmoid(z)
        allZs.append(z)
        allActivations.append(activation)
        layerInput = np.insert(activation, 0, np.ones(activation.shape[0]), axis=1)

    #last activation corresponds to the prediction values
    predictions = activation

    #calculate cost
    #first, transform each y into suitable vector of np.zeros and np.ones
    yMatrix = []
    for i in y.reshape(y.size).A[0]:
        tmp = np.linspace(0, numLabels-1, numLabels)
        tmp = tmp == i
        tmp = tmp*1
        yMatrix.append(tmp)
    yMatrix = np.matrix(yMatrix)

    #compute cost
    #numpy.np.multiply returns element-wise multiplication of matrices and np.arrays
    J = np.multiply(yMatrix, np.log(predictions)) + np.multiply(1-yMatrix, np.log(1-predictions))
    J = J.reshape(J.size, order="F").A[0]
    J = -sum(J)/numSamples

    #regularize the cost
    flatTheta = []
    for t in allTheta:
        t = t[:, 1:]
        tmp = np.reshape(t, t.size, order="F").A[0]
        flatTheta.extend(tmp)

    J = J + _lambda*sum(np.multiply(flatTheta, flatTheta))/(2*numSamples)
    print count, J
    return J

def derivatives(nnParams, hiddenLayerSizes,\
         numLabels, X, y, _lambda):
    """
    nnParams: Unrolled theta/weights.
    hiddenLayerSizes: Number of units in each hidden layer. It should be an
    np.array/list.
    numLabels: Number of class labels, which is same as number of units in
    output layer.
    X: feature np.matrix, must NOT include np.ones as first column.
    y: column np.matrix of labels, each of them should be an integer between 0 and
    numLabels.
    _lambda: regularization parameter.

    """
    inputLayerSize = X.shape[1]
    numHiddenLayers = len(hiddenLayerSizes)

    #roll nnParams into thetas based on layer sizes 
    allTheta = [] #np.array of theta matrices

    layerSizes = []
    layerSizes.extend(hiddenLayerSizes)
    layerSizes.append(numLabels)

    prevLayerSize = inputLayerSize
    startIndex = 0
    for layerNum in xrange(numHiddenLayers+1):
        curLayerSize = layerSizes[layerNum]
        numValues = curLayerSize * (prevLayerSize+1)
        endIndex = startIndex+numValues
        theta = nnParams[startIndex:endIndex]
        theta = theta.reshape(curLayerSize, prevLayerSize+1, order="F")
        theta = np.matrix(theta)
        allTheta.append(theta)
        startIndex=endIndex
        prevLayerSize = curLayerSize

    numSamples = X.shape[0]
    
    #compute activations using thetas and inputs
    allActivations = []
    allZs = []
    allZs.append(np.matrix([])) #dummy matrix to keep indexing easy for backprop
    #use numpy.np.insert to add column of np.ones!
    #np.insert(np.matrix/arr, indexToInsertBefore, axis=0(row)/1(column))
    layerInput = np.insert(X, 0, np.ones(numSamples), axis=1)
    allActivations.append(X) #first layer activations are inputs

    for layerNum in xrange(numHiddenLayers+1):
        theta = allTheta[layerNum]
        z = layerInput*theta.T
        activation = sigmoid(z)
        allZs.append(z)
        allActivations.append(activation)
        layerInput = np.insert(activation, 0, np.ones(activation.shape[0]), axis=1)

    #calculate the gradient!
    #first, transform each y into suitable vector of np.zeros and np.ones
    yMatrix = []
    for i in y.reshape(y.size).A[0]:
        tmp = np.linspace(0, numLabels-1, numLabels)
        tmp = tmp == i
        tmp = tmp*1
        yMatrix.append(tmp)
    yMatrix = np.matrix(yMatrix)

    #back propagation to compute partial derivatives
    allDerivatives = []
    for i in xrange(numHiddenLayers+1):
        allDerivatives.append(np.matrix([]))

    #delta for output layer is just activations-y, and from it the other deltas are computed
    delta = allActivations[-1]-yMatrix
    deltaNextLayer = delta

    #Imagine input layer is layer0, hidden layers 1,2.. and output layer n-1.
    #allActivations[0] are activations of first layer
    #allZs[0] is allActivations[0]*allTheta[0]
    _range = np.array(range(numHiddenLayers+1))+1
    for layerNum in reversed(_range):
        activations = allActivations[layerNum-1]
        activations = np.insert(activations, 0, np.ones(activations.shape[0]), axis=1)
        derivative = deltaNextLayer.T*activations
        derivative = derivative/numSamples

        #add regularization
        theta = allTheta[layerNum-1]
        firstCol = derivative[:, 0]
        firstCol = firstCol.reshape(firstCol.size, order="F")
        derivative = derivative[:, 1:] + (_lambda/numSamples)*theta[:, 1:]
        derivative = np.insert(derivative, 0, firstCol, axis=1)

        allDerivatives[layerNum-1] = derivative

        #compute delta of this layer and move on to previous layer
        #and remember we don't need delta for input layer, so...
        if layerNum == 1:
            break

        delta = deltaNextLayer*theta
        delta = delta[:, 1:]
        delta = np.multiply(delta, sigmoidGradient(allZs[layerNum-1]))

        #allDeltas[layerNum] = delta
        deltaNextLayer = delta


    #unroll partial derivative terms
    grad = np.array([])
    for D in allDerivatives:
        grad = np.append(grad, D.reshape(D.size, order="F").A[0], axis=0)

    return grad

def predict(allTheta, X):
    """
    Given a trained neural network (i.e., weights/thetas), this function outputs
    labels for given data in X
    allTheta: np.array of matrices.
    X: input data, without bias terms/np.ones in first column.
    """
    numSamples = X.shape[0]
    _input = np.insert(X, 0, np.ones(numSamples), axis=1)
    for theta in allTheta:
        result = sigmoid(_input*theta.T)
        _input = np.insert(result, 0, np.ones(numSamples), axis=1)

    predictions = np.argmax(result, axis=1)
    return predictions

def train(X, y, hiddenLayerSizes,\
          numLabels, _lambda):
    """
    X: input data matrix
    y: labels corresponding to data samples in X
    """
    inputLayerSize = X.shape[1]

    initialTheta = []
    prevLayerSize = inputLayerSize
    for size in hiddenLayerSizes:
        weights = randInitializeWeights(prevLayerSize, size)
        initialTheta.extend(weights)
        prevLayerSize = size
    weights = randInitializeWeights(prevLayerSize, numLabels)
    initialTheta.extend(weights)

    funcHandle = lambda theta: cost(theta, hiddenLayerSizes, numLabels, X, y,\
                                    _lambda)
    funcDerivativeHandle = lambda theta: derivatives(theta, hiddenLayerSizes, numLabels, X, y,\
                                    _lambda)
    res = fmin_cg(funcHandle, initialTheta, fprime=funcDerivativeHandle,\
                    maxiter=50)

    #roll back theta values and return an allTheta!
    allTheta = [] #np.array of theta matrices

    layerSizes = []
    layerSizes.extend(hiddenLayerSizes)
    layerSizes.append(numLabels)
    numHiddenLayers = len(hiddenLayerSizes)

    prevLayerSize = inputLayerSize
    startIndex = 0
    for layerNum in xrange(numHiddenLayers+1):
        curLayerSize = layerSizes[layerNum]
        numValues = curLayerSize * (prevLayerSize+1)
        endIndex = startIndex+numValues
        theta = res[startIndex:endIndex]
        theta = theta.reshape(curLayerSize, prevLayerSize+1, order="F")
        theta = np.matrix(theta)
        allTheta.append(theta)
        startIndex=endIndex
        prevLayerSize = curLayerSize

    return allTheta

def numGrad(theta, hiddenLayerSizes, numLabels, X, y, _lambda):
    numgrad = np.zeros(theta.size);
    perturb = np.zeros(theta.size);
    e = 1e-4;
    for p in xrange(theta.size):
        perturb[p] = e;
        loss1 = cost(theta - perturb, hiddenLayerSizes, numLabels, X, y, _lambda);
        loss2 = cost(theta + perturb, hiddenLayerSizes, numLabels, X, y, _lambda);
        numgrad[p] = (loss2 - loss1) / (2*e);
        perturb[p] = 0;
    return numgrad

def check(initialTheta, X, y, hiddenLayerSizes, numLabels, _lambda):
#    inputLayerSize = X.shape[1]
#    initialTheta = []
#    prevLayerSize = inputLayerSize
#    for size in hiddenLayerSizes:
#        weights = randInitializeWeights(prevLayerSize, size)
#        initialTheta.extend(weights)
#        prevLayerSize = size
#    weights = randInitializeWeights(prevLayerSize, numLabels)
#    initialTheta.extend(weights)
    
    c = cost(initialTheta, hiddenLayerSizes, numLabels, X, y, _lambda)
    d = derivatives(initialTheta, hiddenLayerSizes, numLabels, X, y, _lambda)
    #nd = numGrad(initialTheta, hiddenLayerSizes, numLabels, X, y, _lambda)

    return [c,d]

if __name__ == "__main__":
    X = np.loadtxt("smallX.txt", delimiter=",")
    X = np.matrix(X)
    y = np.loadtxt("smally.txt")
    y = np.matrix(y)
    inputLayerSize = X.shape[1]
    hiddenLayerSizes = [3]
    numLabels = 10
    _lambda = 0.1

    initialTheta = []
    prevLayerSize = inputLayerSize
    for size in hiddenLayerSizes:
        weights = randInitializeWeights(prevLayerSize, size)
        initialTheta.extend(weights)
        prevLayerSize = size
    weights = randInitializeWeights(prevLayerSize, numLabels)
    initialTheta.extend(weights)

    funcHandle = lambda theta: cost(theta, hiddenLayerSizes, numLabels, X, y,\
                                    _lambda)
    funcDerivativeHandle = lambda theta: derivatives(theta, hiddenLayerSizes, numLabels, X, y,\
                                    _lambda)
    res = fmin_cg(funcHandle, initialTheta, fprime=funcDerivativeHandle,\
                    maxiter=50)
