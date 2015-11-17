# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 11:06:09 2015

@author: dlituiev
"""

from sklearn.linear_model.poisson_regression import PoissonRegression

import importlib
# importlib.reload(sklearn)
import numpy as np
import matplotlib.pyplot as plt

def simulator(theta = 10, nsamples = 5000, xmean = .1, 
              xdistribution = np.random.randn):
    
    nfeatures = 1 if not hasattr(theta, "shape") else theta.shape
    nfeatures = 1 if not  hasattr(theta, "__len__") else len(theta)
    X = xdistribution( nsamples, nfeatures )*xmean
    
    #print("X:", X.shape)
    #if nfeatures == 1:
    #    X = X.reshape(-1, 1)
    
    lambda_ = np.exp(X.dot(theta))
    #X = X[(lambda_>=0).all(1), :]
    #lambda_ = lambda_[(lambda_>=0).all(1), :]
    
    y = np.random.poisson( lambda_ )
    return X, y

X, y = simulator(theta = [1,2,3], xmean = .10 )

pr = PoissonRegression()
theta_hat = pr.fit(X, y, theta0 = .7, maxiter=2000)
print("newton theta:", theta_hat)



plt.hist(y, 50)
pass

"how log-likelihood looks like?"
theta = np.linspace(0,10,1000)

def loglh(X, y, theta):
    return np.sum( y*X*theta - np.exp( theta * X ) , 0)
    
negloglh = -loglh( X, y ,theta)

plt.plot(theta, negloglh)

print("argmin(l(theta))", theta[np.argmin(negloglh)] )

#np.sum( y*X -  X * np.exp( theta * X ) , 0)
#################

np.log(np.mean(y))

lambda_ = theta*X
np.exp(lambda_)

"""
1. 

"""