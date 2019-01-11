from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import pykalman


class ConstAccelKalmanFilter(object):
    """
    Implements a Kalman Filter with a constant acceleration model for offline filtering and 
    smoothing. 
    """

    def __init__(self,dt,qval=1.0,rval=1.0):

        self.dt = dt
        self.qval = qval
        self.rval = rval

    def smooth(self,data):
        kalman_filter = self.get_kalman_filter(data[0,:])
        data_filt, covariance_filt = kalman_filter.smooth(data)
        return data_filt, covariance_filt

    def filter(self,data):
        kalman_filter = self.get_kalman_filter(data[0,:])
        data_filt, covariance_filt = kalman_filter.filter(data)
        return data_filt, covariance_filt

    def get_kalman_filter(self,x0):
        try:
            ndim = x0.shape[0]
        except AttributeError:
            ndim = len(x0)

        kalman_filter = pykalman.KalmanFilter(
            transition_matrices = get_state_transition_matrix(ndim,self.dt),
            observation_matrices = get_observation_matrix(ndim),
            transition_covariance = get_state_transition_covariance_matrix(ndim,self.dt,self.qval),
            observation_covariance = get_observation_covariance_matrix(ndim,self.rval),
            initial_state_mean = get_initial_state_mean(x0),
            initial_state_covariance = get_initial_state_covariance_matrix(ndim),
                )
        return kalman_filter
    

# Utility Functions
# ---------------------------------------------------------------------------------------
def get_state_transition_matrix(ndim,dt):
    A = np.eye(3*ndim)
    for i in range(2*ndim):
        A[i,i+ndim] = dt
    for i in range(ndim):
        A[i,i+2*ndim] = 0.5*dt**2
    return A

def get_observation_matrix(ndim):
    C = np.zeros((ndim, 3*ndim))
    C[:ndim,:ndim] = np.eye(ndim)
    return C

def get_state_transition_covariance_matrix(ndim,dt,qval):
    Q = np.eye(3*ndim)
    for i in range(ndim):
        Q[i,i] = 1.0 
        Q[i+ndim, i+ndim] = 1.0/dt**2
        Q[i+2*ndim, i+2*ndim] = 1.0/(dt**4)
    Q = qval*Q
    return Q

def get_observation_covariance_matrix(ndim,rval):
    R = rval*np.eye(ndim)
    return R

def get_initial_state_mean(x0):
    return np.array([x0[0], x0[1], 0.0, 0.0, 0.0, 0.0])

def get_initial_state_covariance_matrix(ndim):
    return np.eye(3*ndim)

