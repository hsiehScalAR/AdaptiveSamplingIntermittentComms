# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from .kern import Kern
from ...core.parameterization import Param
from paramz.transformations import Logexp
import numpy as np

class SpatioTemporal(Kern):
    """
    SpatioTemporal kernel

    :param input_dim: the number of input dimensions
    :type input_dim: int
    :param variance:
    :type variance: float
    """
    def __init__(self, input_dim=3, variance=1., a=1., b=1., active_dims=None, name='SpatioTemporal'):
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
        if X2 is None: 
            print('X2 is None')
            X2 = X
        x = np.array(X[:,0:2]).reshape((-1,2))
        x2 = np.array(X2[:,0:2]).reshape((-1,2))
        t = np.array(X[:,2]).reshape((-1,1))
        t2 = np.array(X2[:,2]).reshape((-1,1))
        
        # print((x+x2))
        # print(np.dot(x,x2.T))
        
        # TODO: Problem when inferring because X and X2 have completly different shapes.
        #  Maybe look at example from linear with dot product
        #  It seems like X2 is always None unless we infer
        #  Also need to include norm again
        
        first = self.variance**2*np.exp(-self.b**2*np.dot((x+x2),(x+x2).transpose())-self.a**2*np.dot((t+t2),(t+t2).transpose()))
        second = self.variance**2*np.exp(-self.b**2*np.dot((x-x2),(x-x2).transpose())-self.a**2*np.dot((t-t2),(t-t2).transpose()))
        third = -2*(self.variance**2*np.exp(-self.b**2*np.dot(x,x.transpose())-self.a**2*np.dot(t,t.transpose())) + self.variance**2*np.exp(-self.b**2*np.dot(x2,x2.transpose())-self.a**2*np.dot(t2,t2.transpose()))-self.variance**2)
        
        cov = first + second + third

        return cov

    def Kdiag(self,X):
        x = np.array(X[:,0:2]).reshape((-1,2))
        x2 = x
        t = np.array(X[:,2]).reshape((-1,1))
        t2 = t
        
        first = self.variance**2*np.exp(-self.b**2*np.dot((x+x2),(x+x2).transpose())-self.a**2*np.dot((t+t2),(t+t2).transpose()))
        second = self.variance**2*np.exp(-self.b**2*np.dot((x-x2),(x-x2).transpose())-self.a**2*np.dot((t-t2),(t-t2).transpose()))
        third = -2*(self.variance**2*np.exp(-self.b**2*np.dot(x,x.transpose())-self.a**2*np.dot(t,t.transpose())) + self.variance**2*np.exp(-self.b**2*np.dot(x2,x2.transpose())-self.a**2*np.dot(t2,t2.transpose()))-self.variance**2)
        
        cov = first + second + third

        return np.diag(cov)
    
    def update_gradients_full(self, dL_dK, X, X2):
        if X2 is None: X2 = X

        x = np.array(X[:,0:2]).reshape((-1,2))
        x2 = np.array(X2[:,0:2]).reshape((-1,2))
        t = np.array(X[:,2]).reshape((-1,1))
        t2 = np.array(X2[:,2]).reshape((-1,1))

        first = np.exp(-self.b**2*np.dot((x+x2),(x+x2).transpose())-self.a**2*np.dot((t+t2),(t+t2).transpose()))
        second = np.exp(-self.b**2*np.dot((x-x2),(x-x2).transpose())-self.a**2*np.dot((t-t2),(t-t2).transpose()))
        third1 = np.exp(-self.b**2*np.dot(x,x.transpose())-self.a**2*np.dot(t,t.transpose()))
        third2 = np.exp(-self.b**2*np.dot(x2,x2.transpose())-self.a**2*np.dot(t2,t2.transpose()))

        dvar = 2*self.variance*(first + second -2*(third1 + third2 - 1)) 
        da = -2*self.a*self.variance**2*(first + second) +4*self.a*self.variance**2*(third1 + third2) 
        db = -2*self.b*self.variance**2*(first + second) +4*self.b*self.variance**2*(third1 + third2)
        
        self.variance.gradient = np.sum(dvar*dL_dK)
        self.a.gradient = np.sum(da*dL_dK)
        self.b.gradient = np.sum(db*dL_dK)
    
    # def update_gradients_diag(self, dL_dKdiag, X):
    #     self.variance.gradient = np.sum(dL_dKdiag)
    #     # here self.lengthscale and self.power have no influence on Kdiag so target[1:] are unchanged
    
    # def gradients_X(self,dL_dK,X,X2):
    #     """derivative of the covariance matrix with respect to X."""
    #     if X2 is None: X2 = X
    #     dist2 = np.square((X-X2.T)/self.lengthscale)
    
    #     dX = -self.variance*self.power * (X-X2.T)/self.lengthscale**2 *  (1 + dist2/2./self.lengthscale)**(-self.power-1)
    #     return np.sum(dL_dK*dX,1)[:,None]
    
    # def gradients_X_diag(self,dL_dKdiag,X):
    #     # no diagonal gradients
    #     pass


