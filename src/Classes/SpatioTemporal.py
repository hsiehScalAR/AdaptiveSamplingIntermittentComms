# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from .kern import Kern
from ...core.parameterization import Param
from paramz.transformations import Logexp
from ...util.linalg import tdot
from ... import util

import numpy as np

class SpatioTemporal(Kern):
    """
    SpatioTemporal kernel

    :param input_dim: the number of input dimensions
    :type input_dim: int
    :param variance:
    :type variance: float
    """
    # TODO: figure out why it does not update the parameters when I set them to something different than 1 initially
    def __init__(self, input_dim=3, variance=3., a=4., b=50., active_dims=None, name='SpatioTemporal'):
        assert input_dim==3
        super(SpatioTemporal, self).__init__(input_dim, active_dims, name)
        
        self.variance = Param('variance', variance, Logexp())
        self.a = Param('a', a, Logexp())
        self.b = Param('b', b, Logexp())
        
        self.link_parameters(self.variance, self.a, self.b)
        
    def parameters_changed(self):
        # nothing todo here
        pass
    
    def K(self,X,X2):     
        x = np.array(X[:,0:2]).reshape((-1,2))
        t = np.array(X[:,2]).reshape((-1,1))
        if X2 is None: 
            x2 = None
            t2 = None
        else:
            x2 = np.array(X2[:,0:2]).reshape((-1,2))
            t2 = np.array(X2[:,2]).reshape((-1,1))

        xpos = self.distPos(self.b,x,x2)
        tpos = self.distPos(self.a,t,t2)
        xneg = self.distNeg(self.b,x,x2)
        tneg = self.distNeg(self.a,t,t2)



        first = self.variance**2*np.exp(-xpos**2-tpos**2)
        second = self.variance**2*np.exp(-xneg**2-tneg**2)

        xx = self.diagNorm(self.b,x)
        tt = self.diagNorm(self.a,t)
        if x2 is None:
            xx2 = np.ones_like(xx)
            tt2 = np.ones_like(tt)
        else:
            xx2 = self.diagNorm(self.b,x2)
            tt2 = self.diagNorm(self.a,t2)

        third = (-2*(np.dot(self.variance**2*np.exp(-xx**2-tt**2), self.variance**2*np.exp(-xx2**2-tt2**2).T)-self.variance**2))
        
        cov = first + second + third

        return cov

    def Kdiag(self,X):
        # ret = np.empty(X.shape[0])
        # ret[:] = self.variance
        # return ret
        x = np.array(X[:,0:2]).reshape((-1,2))
        t = np.array(X[:,2]).reshape((-1,1))
        x2 = None
        t2 = None
        
        # xpos = self.distSingle(self.b,x)
        # tpos = self.distSingle(self.a,t)
        # xneg = self.distSingle(self.b,x)
        # tneg = self.distSingle(self.a,t)
        
        xx = self.diagNorm(self.b,x)
        tt = self.diagNorm(self.a,t)
        xx2 = self.diagNorm(self.b,x2)
        tt2 = self.diagNorm(self.a,t2)
        
        # first and second are zero on diagonal
        # first = self.variance**2*np.exp(-xpos**2-tpos**2)

        # second = self.variance**2*np.exp(-xneg**2-tneg**2)

        third = (-2*(self.variance**2*np.exp(-xx**2-tt**2) + self.variance**2*np.exp(-xx2**2-tt2**2)-self.variance**2))

        # cov = first + second + third
        cov = third

        return np.reshape(cov,(-1))

    def update_gradients_full(self, dL_dK, X, X2):
        x = np.array(X[:,0:2]).reshape((-1,2))
        t = np.array(X[:,2]).reshape((-1,1))
        if X2 is None: 
            x2 = None
            t2 = None
        else:
            x2 = np.array(X2[:,0:2]).reshape((-1,2))
            t2 = np.array(X2[:,2]).reshape((-1,1))

        xpos = self.distPos(self.b,x,x2)
        tpos = self.distPos(self.a,t,t2)
        xneg = self.distNeg(self.b,x,x2)
        tneg = self.distNeg(self.a,t,t2)

        xx = self.diagNorm(self.b,x)
        tt = self.diagNorm(self.a,t)
        xx2 = self.diagNorm(self.b,x2)
        tt2 = self.diagNorm(self.a,t2)

        first = np.exp(-xpos**2-tpos**2)
        second = np.exp(-xneg**2-tneg**2)
        third1 = np.exp(-xx**2-tt**2)
        third2 = np.exp(-xx2**2-tt2**2)

        dvar = 2*self.variance*(first + second -2*(np.dot(third1, third2.T) - 1)) 
        da = -2*self.a*self.variance**2*(first + second) +4*self.a*self.variance**2*np.dot(third1, third2.T) 
        db = -2*self.b*self.variance**2*(first + second) +4*self.b*self.variance**2*np.dot(third1, third2.T)
        
        self.variance.gradient = np.sum(dvar*dL_dK)
        self.a.gradient = np.sum(da*dL_dK)
        self.b.gradient = np.sum(db*dL_dK)
    
    def distNeg(self, lengthscale, X, X2=None):
        """
        Compute the Euclidean distance between each row of X and X2, or between
        each pair of rows of X if X2 is None.
        """
        if X2 is None:
            Xsq = np.sum(np.square(X),1)
            r2 = -2.*tdot(X) + (Xsq[:,None] + Xsq[None,:])
            util.diag.view(r2)[:,]= 0. # force diagnoal to be zero: sometime numerically a little negative
            # r2 = np.clip(r2, 0, np.inf)
            return np.sqrt(r2)/lengthscale
        else:
            X1sq = np.sum(np.square(X),1)
            X2sq = np.sum(np.square(X2),1)
            r2 = -2.*np.dot(X, X2.T) + (X1sq[:,None] + X2sq[None,:])
            # r2 = np.clip(r2, 0, np.inf)
            return np.sqrt(r2)/lengthscale

    def distPos(self, lengthscale, X, X2=None):
        """
        Compute the Euclidean distance between each row of X and X2, or between
        each pair of rows of X if X2 is None.
        """
        if X2 is None:
            Xsq = np.sum(np.square(X),1)
            r2 = +2.*tdot(X) + (Xsq[:,None] + Xsq[None,:])
            util.diag.view(r2)[:,]= 0. # force diagnoal to be zero: sometime numerically a little negative
            # r2 = np.clip(r2, 0, np.inf)
            return np.sqrt(r2)/lengthscale
        else:
            X1sq = np.sum(np.square(X),1)
            X2sq = np.sum(np.square(X2),1)
            r2 = +2.*np.dot(X, X2.T) + (X1sq[:,None] + X2sq[None,:])
            # r2 = np.clip(r2, 0, np.inf)
            return np.sqrt(r2)/lengthscale
    
    def distSingle(self, lengthscale, X):
        """
        Compute the Euclidean distance between each row of X and X2, or between
        each pair of rows of X if X2 is None.
        """
        if X is None:
            return np.array([0])
        # r2 = np.sum(np.square(X))
        r2 = np.matmul(X,X.T)
        # r2 = np.clip(r2, 0, np.inf)
        return np.sqrt(r2)/lengthscale

    def diagNorm(self, lengthscale, X):
        """
        Compute the Euclidean distance between each row of X and X2, or between
        each pair of rows of X if X2 is None.
        """
        if X is None:
            return 0
        r2 = np.sum(np.square(X),1)
        # r2 = np.clip(r2, 0, np.inf)
        return np.reshape(np.sqrt(r2)/lengthscale,(-1,1))



