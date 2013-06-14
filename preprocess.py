from __future__ import division
import numpy as np
import random

def featureNormalize(X):
    """
    X: a feature matrix with one row per sample (without bias unit)

    returns [X_norm, mu, sigma] where mu and sigma have means and standard
    deviations of each feature in X.
    """
    X_norm = np.zeros_like(X)
    numFeatures = X.shape[1]
    mu = np.zeros(numFeatures)
    sigma = np.zeros(numFeatures)
    for i in xrange(numFeatures):
        mu[i] = np.mean(X[:, i])
        sigma[i] = np.std(X[:, i])
        if sigma[i] != 0:
            X_norm[:, i] = (X[:, i]-mu[i])/sigma[i]

    return [X_norm, mu, sigma]

def subSample(dataMatrix, maxSamples=0):
    """
    dataMatrix is a matrix with each row being a sample, along with the class label
    (integer) as the last element in each row. The class labels are assumed to
    be from 0 to numClasses-1.
    maxSamples is the maximum number of samples allowed per class. When it is
    passed as 0, then it is set to the number of samples the smallest class has. 

    Returns sampleMatrix where number of samples for all classes is equal.
    """

    #assumes class indices are from 0 to numClasses-1
    numClasses = max(dataMatrix[:, -1])+1
    classIndices = {}

    for c in xrange(numClasses):
        res = np.where(dataMatrix[:, -1] == c)
        classIndices[c] = list(res[0].A[0])

    if maxSamples == 0:
        #determine number of samples in the smallest class
        counts = [len(classIndices[c]) for c in xrange(numClasses)]
        maxSamples = min(counts)

    rnd = random.Random()
    #chose not to pass the seed, as python's random chooses system's current
    #time or operating system provided mechanism to seed automatically
    
    selectedIndices = []
    for c in xrange(numClasses):
        rnd.shuffle(classIndices[c])
        selectedIndices.extend(classIndices[c][:maxSamples])

    #selectedIndices = np.array(selectedIndices)
    return dataMatrix[selectedIndices, :]


def splitData(dataMatrix, percentages=[60, 20, 20]):
    """
    dataMatrix is a matrix with each row being a sample, along with the class label
    (integer) as the last element in each row. 
    percetanges: a list of percentages indicating [train, validation, test]
    percetange splits.
    
    Returns [train, validation, test] sets.

    """
    if sum(percentages) != 100:
        print "Sorry, no magic wands at work at the moment. Percentages must sum up to 100!"
        return [[], [], []]
    rnd = random.Random()
    #chose not to pass the seed, as python's random chooses system's current
    #time or operating system provided mechanism to seed automatically

    numSamples = dataMatrix.shape[0]
    numClasses = int(max(dataMatrix[:, -1])+1)

    splitIndices = [[], [], []]

    for c in xrange(numClasses):
        res = np.where(dataMatrix[:, -1] == c)
        classIndices = list(res[0].A[0])
        rnd.shuffle(classIndices)
        numClassSamples = len(classIndices)
        startIndex = 0
        for i in xrange(len(percentages)):
            endIndex = startIndex+int(numClassSamples*percentages[i]/100)
            splitIndices[i].extend(classIndices[startIndex:endIndex])
            startIndex = endIndex

    return [dataMatrix[splitIndices[0], :], dataMatrix[splitIndices[1], :],\
            dataMatrix[splitIndices[2], :]]
