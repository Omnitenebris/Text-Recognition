import math
import pickle
import random
import bonnerlib2
import numpy as np
import numpy.random as rnd
import sklearn.utils as skl
import matplotlib.pyplot as plt
import sklearn.linear_model as lin
 
with open('mnist.pickle','rb') as f: data = pickle.load(f)
 
def genData(mu0, mu1, N):
    #Ensure mu are 1-dimensional vectors
    mu0 = np.reshape(mu0, [-1])
    mu1 = np.reshape(mu1, [-1])
   
    #Generate empty dimensional vectors
    mu00 = np.full((N,1), mu0[0])
    mu01 = np.full((N,1), mu0[1])
    mu10 = np.full((N,1), mu1[0])
    mu11 = np.full((N,1), mu1[1])
   
    #Add the two based on where it belongs
    mu0V = np.concatenate((mu00, mu01), axis = 1)
    mu1V = np.concatenate((mu10, mu11), axis = 1)
   
    #Generate data for clusters (found on stackoverflow)
    cu0 = rnd.multivariate_normal(np.ones(2), np.diag(np.ones(2)), N)
    cu1 = rnd.multivariate_normal(np.ones(2), np.diag(np.ones(2)), N)  
   
    #Create two empty matrices
    mu0Matrix = np.full((N, 2), mu0V)
    mu1Matrix = np.full((N, 2), mu1V)
   
    #Add to the clusters
    cu0 += mu0Matrix
    cu1 += mu1Matrix
   
    #Generate two dimensional vectors for t
    temp0 = np.full((N,1), int(0))
    temp1 = np.full((N,1), int(1))
   
    #Add the temporary to the clusters based on which cluster (0/1)
    cu0 = np.concatenate((cu0, temp0), axis=1)
    cu1 = np.concatenate((cu1, temp1), axis=1)
    X = np.concatenate((cu0, cu1), axis=0)
   
    #Shuffling to distribute randomly
    t = X[:,2]
    X = X[:,[0,1]]
    t = t.astype(int)
    X, t = skl.shuffle(X, t, random_state=0)
 
    return(X, t)
 
def graphData():
    N = 10000
    mu0 = (0.5, -0.5)
    mu1 = (-0.5, 0.5)
    X, t = genData(mu0, mu1, N)
    colors = np.array(['r', 'b'])
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], color = colors[t], s = 10)
 
#graphData()
   
def logregDemo(N, betaList):
    #Initialize everything
    i = 0
    aList = []
    u0 = (2, -2)
    u1 = (-2, 2)
    bounds = int(math.ceil(np.sqrt(len(betaList))))
   
    #Initalize figure 2
    plt.figure(0)
    plt.suptitle("Figure 2: contour plots of logistic decision functions")
 
    #Initialize figure 3
    plt.figure(1)
    plt.suptitle("Figure 3: surface plots of logistic decision functions")
 
    #Get clf
    clf = lin.LogisticRegression()
   
    #Loop through betaList
    while i < len(betaList):
        #Generate X, t
        index0 = (betaList[i] * u0[0],  betaList[i] * u0[1])
        index1 = (betaList[i] * u1[0],  betaList[i] * u1[1])
        X, t = genData(index0, index1, N)
       
        #Fit the data and get the accuracies
        clf.fit(X, t)
        aList.append(clf.score(X, t))
       
        #Plot figure 1
        figure1 = plt.figure(0).add_subplot(bounds, bounds, i + 1)
        colors = np.array(["r", "b"])
        plt.scatter(X[:, 0], X[:, 1], color = colors[t],s = 0.1)
        bonnerlib2.dfContour(clf, figure1)
       
        #Plot figure 2
        figure2 = plt.figure(1).add_subplot(bounds, bounds, i + 1, projection="3d")
        plt.xlim(-9, 6)
        plt.ylim(-6, 9)
        bonnerlib2.df3D(clf, figure2)
       
        i += 1
       
    return(aList)
 
def runQ2():
    N = 10000
    betaList = [0.1,0.2,0.5,1.0]
    logregDemo(N,betaList)
 
#runQ2()
   
def displaySample(N, D):
    #Initalize counter, figure, and dimensions
    i = 0
    plt.figure()
    plt.suptitle("test")
    bounds = int(math.ceil(np.sqrt(N)))
   
    while i < N:
        #Get subplot and turn off axises
        plt.subplot(bounds, bounds, i + 1)
        plt.axis('off')
       
        #Choose a random integer
        disp = D[random.randint(0,9)]
       
        #Get dimensions
        squareR = int(np.sqrt(disp.shape[1]))
       
        #Reshape random number, reshape it, then plot it
        temp = disp[random.randint(0, len(D))].reshape(squareR, squareR)
        plt.imshow(temp, cmap='Greys', interpolation='nearest')
       
        i += 1
    plt.show()
 
#displaySample(23, data['training'])
 
def flatten(data):
    #Initalize empty variables and list
    i = 0
    X = []
    t = []
   
    #For possible numbers 0-9
    while i < 10:
        #For all possible versions of each integer, append to X,t
        for j in data[i]:
            X.append(j)
            t.append(i)
        i+= 1
   
    #Shuffle both together
    X, t = skl.shuffle(X, t, random_state=0)
    X = np.array(X)
    t = np.array(t)
    return(X, t)
 
#(flatten(data['training'])[1][0])
 
def accuracyCalculator():
    #Initialize variables and lists
    c = 0
    i = 0
    j = 0
    mProb = []
    miTrain= []
 
    #Get training and testing data
    testX, testT = flatten(data['testing'])
    trainX, trainT = flatten(data['training'])
 
    #Code from assignment
    clf = lin.LogisticRegression(multi_class='multinomial', solver='lbfgs')
   
    #Fit and get scores
    clf.fit(trainX, trainT)
    trainPredict = clf.predict(trainX)
    testScore = clf.score(testX, testT)
    trainScore = clf.score(trainX, trainT)
 
    #Get prob and misclassified training data
    tProb = clf.predict_proba(testX)
    print(tProb[0])
    mIT = np.zeros((len(trainX), len(trainX[0])), dtype = np.uint8)
   
    while i < len(trainX):
        #Check if misclassified and mark then count it
        if trainPredict[i] != trainT[i]:
            miTrain.append(trainPredict[i])
            mIT[c] = trainX[i]
            c += 1
        i += 1
   
    #Get max probability for each one
    while j < len(tProb):
        mProb.append(max(tProb[j]))
        j += 1
   
    #Sort it for return
    mProb = np.array(mProb)
    print(mProb)
    print(len(mProb))
    temp = mProb.argsort()
    
    sortTest = testX[temp]
    return trainScore, testScore, mIT[0:c],sortTest
 
acc = accuracyCalculator()
print('Training accuracy is {} and Testing accuracy is {}.'.format(acc[0],acc[1]))
