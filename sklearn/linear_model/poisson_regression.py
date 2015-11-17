from __future__ import division
from abc import ABCMeta, abstractmethod
import numbers
import warnings

import numpy as np
import scipy.sparse as sp
from scipy import linalg
from scipy import sparse

from ..externals import six
from ..externals.joblib import Parallel, delayed
from ..base import BaseEstimator, ClassifierMixin, RegressorMixin
from ..utils import as_float_array, check_array, check_X_y, deprecated
from ..utils import check_random_state, column_or_1d
from ..utils.extmath import safe_sparse_dot
from ..utils.sparsefuncs import mean_variance_axis, inplace_column_scale
from ..utils.fixes import sparse_lsqr
from ..utils.validation import NotFittedError, check_is_fitted
from ..utils.seq_dataset import ArrayDataset, CSRDataset

from .base import LinearModel
from scipy.optimize import newton, fmin_cg, minimize

class PoissonRegression(LinearModel, RegressorMixin):
    """
    Ordinary least squares Linear Regression.
    Parameters
    ----------
    fit_intercept : boolean, optional
        whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (e.g. data is expected to be already centered).
    normalize : boolean, optional, default False
        If True, the regressors X will be normalized before regression.
    copy_X : boolean, optional, default True
        If True, X will be copied; else, it may be overwritten.
    n_jobs : int, optional, default 1
        The number of jobs to use for the computation.
        If -1 all CPUs are used. This will only provide speedup for
        n_targets > 1 and sufficient large problems.
    Attributes
    ----------
    coef_ : array, shape (n_features, ) or (n_targets, n_features)
        Estimated coefficients for the linear regression problem.
        If multiple targets are passed during the fit (y 2D), this
        is a 2D array of shape (n_targets, n_features), while if only
        one target is passed, this is a 1D array of length n_features.
    intercept_ : array
        Independent term in the linear model.
    Notes
    -----
    From the implementation point of view, this is just plain Ordinary
    Least Squares (scipy.linalg.lstsq) wrapped as a predictor object.
    """

    def __init__(self, fit_intercept=True, normalize=False, copy_X=True,
                 n_jobs=1):
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.n_jobs = n_jobs

    def fit(self, X, y, sample_weight=None, theta0 = 1, maxiter=50):
        """
        Fit Poisson regression model.
        Parameters
        ----------
        X :    numpy array or sparse matrix of shape [n_samples,n_features]
            Training data
        y :    numpy array of shape [n_samples, n_targets]
            Target values
        theta0: a vector of initial guess of
                regression coefficients of shape [ n_features , 1]
        Returns
        -------
        self : returns an instance of self.
        """

        n_jobs_ = self.n_jobs
        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc', 'coo'],
                         y_numeric=True, multi_output=True)
        # Theta is a column vector
        # Design matrix (X) has rows as data and columns as predictors
        # lambda needs to be 

        nfeatures = X.shape[1]
        if not hasattr(theta0, "__len__") or len(theta0) != nfeatures:
            theta0 = np.mean( theta0 ) * np.ones( (nfeatures, 1) )

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
            return  np.sum( y*lambda_ - np.exp(lambda_ ) )

        def dloglh(X, y, theta):
            lambda_ = X.dot( theta )
            return  X.T.dot( y -  np.exp( lambda_ ) )

        def d2loglh(X, y , theta):
            lambda_ = X.dot( theta )
            return  -X.T.dot( X * np.exp( lambda_ )[np.newaxis].T )

        negloglh_   = lambda theta : - loglh(X,y, theta)
        negdloglh_  = lambda theta : - dloglh(X, y, theta)
        negd2loglh_ = lambda theta : - d2loglh(X, y, theta)

        res = minimize( negloglh_ , theta0, jac = negdloglh_, hess = negd2loglh_ , method = "Newton-CG")
        theta_hat = res.x
        return theta_hat 
