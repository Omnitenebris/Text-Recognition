import numpy as np
import time
import numpy.random as rnd
import numpy.lingalg as la
import pickle as pickle
import matplotlib.pyplot as plt
import sklearn.linear_model as lin

#Multiply matrices A and B 
def mymult(A,B):
    I,K = np.shape(A)
    K,J = np.shape(B)
    c = np.zeros([I,J])
    for i in range(I):
        for j in range(J):
            #compute the value of c[i,j]
            x = 0.0
            for k in range(K):
                x = x + A[i,k]*B[k,j]
            c[i,j] = x
        return c
    
def mymeasure(I,J,K):
  #create two random matrices
  A = rnd.rand(I,K)
  B = rnd.rand(K,J)
        
  #measure the time to multiply them using numpy.matmul
  t1 = time.time()
  c1 = np.matmul(A,B)
  t2 = time.time()
  print('  ')
  print("Execution time for numpy.matmul( {},{},{}): {}'.format(I,J,K, t2-t1)")
        
  #measure the time to multiply them using mymult
  t1 = time.time()
  c2 =mymult(A,B)
  t2 = time.time()
  print('Execution time for mumult({}, {} {}): {}'.format(I,J,K,t2-t1))
        
  #compute the magnitude of c1-c2
  mag =np.sum((c1-c2)**2)
  print('Magnitude of c1-c2: {}'.format(mag))
        
        
print('\n')
print('-------------')
mymeasure(1000,50,100)
mymeasure(1000,1000,1000)

# read the training and testing data
with open("data1.pickle", 'rb') as f:
    dataTrain,dataTest = pickle.load(f)
        
#set local variable 
Xtrain = dataTrain[:,0]
Ytrain = dataTrain[:,1]
Xtest = dataTest[:,0]
Ytest = dataTest[:,1]
Ntrain = len(Ytrain)
Ntest = len(Ytest)
        
yMax = 15.0
yMin = -15.0
xMin = 0.0
xMax = 1.0

# Construct a data matrix, z, storing the powers of x from 0 to M
def dataMatrix(x,M):
    x = np.reshape(x,[-1])   #ensure x is a 1-dimensional vector
    N = len(x)
    Z = np.ones([N,M+1])     #reserve space for Z
    # compute one column of Z at a time
    for n in range(1,M+1):
        Z[:n] = Z[:,n-1]*x
    return Z

#alternative defintion
def dataMatrix(x,M):
    x = np.reshape(x,[-1,1])   #ensure x is a column 
    mList = range(0,M+1)   #list of powers
    return np.power(x,mList)

#compute the mean squared-error of a polynomial on a data set,
#i.e., return the mean squared magnitude of Y-Zw.
def error(Y,Z,w):
    return np.mean((Y - np.matmul(Z,w))**2)

#fit a polynomial of degree M to the data using linear least-squares
def fitPoly(M):
    Ztrain = dataMatrix(Xtrain,M)
    
    #compute the weight vector
    w = np.linalg.lstsq(Ztrain,Ytrain)[0]
    
    #compute trainning error
    errTrain = error(Ytrain,Ztrain,w)
    #compute test error
    Ztest = dataMatrix(Xtest,M)
    errTest = error(Ytest,Ztest,w)
    
    return w,errTrain,errTest

#plot the polynomial defined by the weight vector w
def plotPoly(w):
    w = np.reshape(w,[-1])   #ensure w is 1-dimensional vector
    M = len(w)-1             #degree of polynomial
    x = np.linspace(xMin,xMax,1000)   # 1000 equally spaced points
    Z = dataMatrix(x,M)
    y = np.matmul(Z,w)             #values of the polynomial
    plt.plot(x,y,'r')              #plot the polynomial
    plt.plot(Xtrain,Ytrain,'b.')   #plot the trainning data 
    plt.ylim(yMin,yMax)

#Find the degree of the polynomial that best fits the data
def bestPoly():
    errTrainList = []    #list of training error
    errTestList = []
    errTestMin = np.inf    #the minimum test error so far
    plt.figure()
    plt.suptitle('Best-fitting polynomials of degree M = 0,1,......,15')
    for m in range(15+1):
        #fit a polynomial of degree m to the data
        w,errTrain,errTest = fitPoly(m)
        #record the training and test errors
        errTrainList.append(errTrain)
        errTestList.append(errTest)
        #plot the polynomial
        plt.subplot(4,4,m+1)
        plotPoly(w)
        #keep track of the best-fitting polynomial found so far
        if errTest < errTestMin:
            errTestMin = errTest
            M = m
            w_best = w
            
            
            
    # plot the lists of training and test error
    plt.figure()
    plt.plot(errTrainList, 'b')
    plt.plot(errTestList, 'r')
    plt.ylim(0,250)
    plt.xlabel('polynomial degree')
    plt.ylabel('error of best-fitting polynomial')
    plt.show()
    
    #plot the best-fitting polynomial
    plt.figure()
    plotPoly(w_best)
    plt.title('Best-fitting polynomial (degree ={})'.format(M))
    plt.xlabel('x')
    plt.ylabel('y')
    
    
    #print optimal values
    print('Degree of best-fitting polynmial: {}'.format(M))
    print(' ')
    print('Weight vector of best-fitting polynomial:')
    print(w_best)
    
    #print errors for best-fitting polynomial
    print('')
    print('Training and test error of best-fitting polynomial:')
    print('    {},  {}'.format(errTrainList[M],errTestMin))
    
print('\n')
print('---------------')
print('')
bestPoly()
   
#rest the validation and testing data
with open('data2.pickle','rb') as f:
    dataVal,dataTest = pickle.load(f)
   
#set global variables 
Xval = dataVal[:,0]
Yval = dataVal[:,1]
Xtest = dataTest[:,0]
Ytest = dataTest[:,1]
Nval = len(Yval)
Ntest = len(Ytest)

#fit a polynmoial of degree M to the data using regularized least-squares,
#where alpha is the coefficient of the regularization term.

def fitRegPoly(M,alpha):
    Ztrain = dataMatrix(Xtrain,M)
    
    #use rodge-regression to compute the weight vector
    ridge = lin.Ridge(alpha)
    ridge.fit(Ztrain,Ytrain)
    w = ridge.coef_
    w[0] = ridge.intercept_
    
    #compute training error
    errTrain = error(Ytrain,Ztrain,w)
    #compute validation error
    Zval = dataMatrix(Xval,M)
    errVal = error(Yval,Zval,w)
    
    return w,errTrain,errVal

#Find the best value of alpha, the regularization coefficient
#Use if to fit a polynomial to data using the regulaized least-squares
def bestRegPoly():
    M = 15   #degree of polynomial
    errTrainList = []    #list of training errors
    errValList = []      #list of validation errors
    errValMin = np.Inf   #the minimum validation-error so far
    alphaList =10.0**np.arange(-13,3) #list of alpha values to consider
    plt.figure()
    plt.suptitle('Best-fitting polynmials for log(alpha) = -13,-12,...,1,2')
    for i in range(15+1):
        alpha =alphaList[i]
        #fit a polynmoial of degree M to the data using alpha
        w,errTrain,errVal = fitRegPoly(M,alpha)
        #record the training and validation errors
        errTrainList.append(errTrain)
        errValList.append(errVal)
        #plot thje polynomial 
        plt.subplot(4,4,i+1)
        plotPoly(w)
        #keep track of the best fitting polynmoial found so far
        if errVal < errValMin:
            errValMin = errVal
            I = i
            w_best = w
             
    
    #plot the lists of training and validation error
    plt.figure()
    plt.semilogx(alphaList,errTrainList,'b')
    plt.semilogx(alphaList,errValList)
    plt.title('Training and validation error')
    plt.xlabel('alpha')
    plt.ylabel('error of best-fitting polynomial')
    plt.show()
    
    #plot the best-fitting polynomial
    plt.figure()
    plotPoly(w_best)
    plt.title('Best-fitting polynomial (alpha = {})'.format(alphaList[I]))
    plt.xlabel('x')
    plt.ylabel('y')
    
    
    #print optimal values
    print(' ')
    print('Optimal value of alpha: {}'.format(alphaList[I]))
    print(' ')
    print(w_best)
    
    
    #compute test error
    Ztest = dataMatrix(Xtest,M)
    errTest = error(Ytest,Ztest,w_best)
    
    #print errors for best-fitting polynomial
    print(' ')
    print('Training, validation and test errors of best-fitting polynomial:')
    print('   {}, {},   {}'.format(errTrainList[I],errValMin, errTest))
    
    
print('\n')
print('----------------')
bestRegPoly()

#compute gradient for linear least-squares

def linGrad(Z,t,w):
    t = np.reshape(t,[-1,1])     #ensure t is a column vector
    w = np.reshape(w,[-1,1])     #ensure w is column vector
    yhat = np.matmul(Z,w)
    err = t-yhat
    err = np.reshape(err,[1,-1])  #convert err to a row vector
    grad = -2.0*np.matmul(err,Z)
    return np.reshape(grad,[-1])

#compute gradient for regularized least-squares

def regGrad(Z,t,w,alpha):
    #compute gradient of the regularization term
    grad = 2.0*alpha*w
    grad[0] = 0.0
    # add in the gradient of the linear least-squares term
    grad = grad + linGrad(Z,t,w)
    return grad

#fit a polynomial of degree M to the data using gradient descent

def fitPolyGrad(M,alpha,Irate):
    w = rnd.randn(M+1)         #initialize the wieght vector randomly
    Ztrain = dataMatrix(Xtrain,M)
    Ztest = dataMatrix(Xtest,M)
    errTrainList = []           # list of training errors
    errTestList = []           # list of test errors
    iList = 10**np.array([0,1,2,3,4,5,6,7])  #list of iterations for plotting
    j = 0               #index of current subplot
    plt.figure()
    plt.suptitle('Fitted polynomial as a number of weight-updates increases')
    for i in range(10000000+1):
        grad = regGrad(Ztrain,Ytrain,w,alpha)     #compute gradient
        w = w - Irate*grad      #update the weight vector
        if np.mod(i,1000) == 0:
            # compute and record training error
            Yhat = np.matmul(Ztrain,w)
            errTrain = np.sum((Ytrain - Yhat)**2)/Ntrain
            errTrainList.append(errTrain)
            #compute and record test error
            Yhat = np.matmul(Ztest,w)
            errTest = np.sum((Ytest - Yhat)**2)/Ntest
            errTestList.append(errTest)
        if i == iList[j]:
            # plot the current polynomial 
            j = j + 1
            plt.subplot(3,3,j)
            plotPoly(w)
            
        if np.mod(i,100000) == 0:
            # monitor the execution
            print('iteration {},   Training error = {}'.format(i,errTrain))
            
    
    
    #plot the fitted polynomial
    plt.figure()
    plotPoly(w)
    plt.xlabel('x')
    plt.ylabel('y')
    
    
    # plot the lists of training and test error
    plt.figure()
    plt.plot(errTrainList, 'b')
    plt.plot(errTestList,'r')
    plt.xlabel('Number of iterations (in thousands)')
    plt.ylabel('Error')
    
    
    
    # print final errors and weight vector
    print(' ')
    print('Final training and test errors')
    print('   {},   {}'.format(errTrain, errTest))
    print(' ')
    print(' final weight vector:')
    print(w)
    
    
    
#print('\n')
#print('---------------')
#fitPolyGrad(15,10.0**(-5),0.01)
#print(' ')
