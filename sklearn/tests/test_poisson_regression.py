# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 11:06:09 2015

@author: dlituiev
"""

from sklearn.linear_model.poisson_regression import PoissonRegression

#import importlib
# importlib.reload(sklearn)
import numpy as np
import matplotlib.pyplot as plt

def simulator(theta = 10, nsamples = 5000, xmean = .1, 
              xdistribution = np.random.randn):
    
    nfeatures = 1 if not hasattr(theta, "shape") else theta.shape
    nfeatures = 1 if not  hasattr(theta, "__len__") else len(theta)
    X = xdistribution( nsamples, nfeatures )*xmean
    #if nfeatures == 1:
    #    X = X.reshape(-1, 1)
    lambda_ = X.dot(theta)
    y = np.random.poisson( np.exp(lambda_ ) )
    return X, y

X, y = simulator(theta = [.7], xmean = .10 )

pr = PoissonRegression()
theta_hat = pr.fit(X, y, theta0 = [.3], maxiter=2000)
print("newton theta:", theta_hat)


from scipy.optimize import check_grad


def loglh(X, y, theta):
    """
    X:        nsamples  x nfeatures
    theta:    nfeatures x 1
    lambda_ : nsamples x 1
    y      :  nsamples x 1    
    
    """
    lambda_ = X.dot( theta )
    if len(lambda_.shape) == 1:
        lambda_ = lambda_[np.newaxis].T 
    if len(y.shape) == 1:
        y = y[np.newaxis].T 
    assert lambda_.shape == y.shape, "shape mismatch, y: %s, lambda: %s" % (repr(y.shape), repr(lambda_.shape) )    
    return  np.sum(y*lambda_ - np.exp(lambda_ ) ) #.reshape(-1,1) 
    #return np.sum(y*lambda_)
    #return np.sum(- np.exp(lambda_ ) )

def dloglh(X, y, theta):
    
    lambda_ = X.dot( theta )
    if len(lambda_.shape) == 1:
        lambda_ = lambda_[np.newaxis].T 
    if len(y.shape) == 1:
        y = y[np.newaxis].T 
    assert lambda_.shape == y.shape, "shape mismatch, y: %s, lambda: %s" % (repr(y.shape), repr(lambda_.shape) )
#    print("lambda", lambda_.shape )
#    print("y", y.shape )
    return  X.T.dot( (y -  np.exp( lambda_ ) ) )
    #return X.T.dot(y)
    #return X.T.dot(-np.exp( lambda_ ).reshape(-1,1))

def d2loglh(X, y , theta):
    lambda_ = X.dot( theta )
    print("lambda:", lambda_.shape )
    print( "theta " , theta.shape )
    print( "X * np.exp( lambda_ )",  (X * np.exp( lambda_ )[np.newaxis].T ).shape )

    return  -X.T.dot( X * np.exp( lambda_ )[np.newaxis].T )
            # np.sum( -(1+ X**2 )* np.exp( theta * X ) )

loglh_ = lambda theta : loglh(X,y, theta)
dloglh_ = lambda theta : dloglh(X, y, theta)
d2loglh_ = lambda theta : d2loglh(X, y, theta)

check_grad(loglh_, dloglh_, np.r_[0.7])


check_grad(dloglh_, d2loglh_, np.r_[1,.3])

#plt.hist(y, 50)
#pass
#
"how log-likelihood looks like?"
theta = np.linspace(0,5,20)

negloglh = [ -loglh( X, y.reshape(-1,1) , [th]) for th in theta ]
#
plt.plot(theta, negloglh)
#
print("argmin(l(theta))", theta[np.argmin(negloglh)] )

"for scalar theta:"
class poisson_scalar():
    
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def loglh(self, X, y, theta):
        lambda_ = X*theta 
        return np.sum( y* lambda_ - np.exp( lambda_ ) , 0)
    
    def dloglh(self,X, y, theta):
        lambda_ = X*theta 
        return  np.sum(X*( (y -  np.exp( lambda_ ) ) ))
    
    def d2loglh(self, X, y, theta):
        lambda_ = X*theta 
        return  -np.sum(X**2 * np.exp( lambda_ ) )
    
    def loglh_(self, theta):
        return self.loglh( X.ravel(), self.y, theta)
    
    def dloglh_(self, theta):
        return self.dloglh( self.X.ravel(), self.y, theta)
        
    def d2loglh_(self, theta):
        return self.d2loglh( self.X.ravel(), self.y, theta)

ps = poisson_scalar(X , y )
negloglh  =  - ps.loglh_( theta)

theta0 = 0.9
maxiter = 50
from scipy.optimize import newton, minimize
theta_hat = newton( ps.dloglh_ , theta0, fprime = ps.d2loglh_, maxiter = maxiter)
print("theta_hat", theta_hat)


res = minimize(lambda x: -ps.loglh_(x), theta0, jac = lambda x: -ps.dloglh_(x), 
               hess = lambda x: -ps.d2loglh_(x) , 
               method = "Newton-CG", options={'disp': True})
        ##"L-BFGS-B")
theta_hat = res.x
print("theta_hat", theta_hat)

res = minimize(ps.loglh_, theta0, jac = ps.dloglh_, hess = ps.d2loglh_ ,
               method ="L-BFGS-B",  options={'disp': True})
        ##"L-BFGS-B")
theta_hat = res.x


res = minimize(loglh_, [theta0], jac = dloglh_, hess = ps.d2loglh_ ,
               method ="Nelder-Mead",  options={'disp': True})
        ##"L-BFGS-B")
theta_hat = res.x
        
#
##np.sum( y*X -  X * np.exp( theta * X ) , 0)
##################
#
#np.log(np.mean(y))
#
#lambda_ = theta*X
#np.exp(lambda_)

"""
1. 

"""