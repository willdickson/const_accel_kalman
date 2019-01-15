from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import pykalman


class ConstAccelKalmanFilter(object):
    """
    Implements a Kalman Filter with a constant acceleration model for offline filtering and 
    smoothing. 
    """

    def __init__(self, qval=1.0, rval=1.0):

        self.qval = qval
        self.rval = rval

    def smooth(self, data, dt):
        kalman_filter = self.get_kalman_filter(data[0,:], dt)
        data_filt, covariance_filt = kalman_filter.smooth(data)
        return data_filt, covariance_filt

    def filter(self, data, dt):
        kalman_filter = self.get_kalman_filter(data[0,:], dt)
        data_filt, covariance_filt = kalman_filter.filter(data)
        return data_filt, covariance_filt

    def get_kalman_filter(self, x0, dt):
        try:
            ndim = x0.shape[0]
        except AttributeError:
            ndim = len(x0)

        kalman_filter = pykalman.KalmanFilter(
            transition_matrices = get_state_transition_matrix(ndim,dt),
            observation_matrices = get_observation_matrix(ndim),
            transition_covariance = get_state_transition_covariance_matrix(ndim,dt,self.qval),
            observation_covariance = get_observation_covariance_matrix(ndim,self.rval),
            initial_state_mean = get_initial_state_mean(x0),
            initial_state_covariance = get_initial_state_covariance_matrix(ndim),
                )
        return kalman_filter
    

# Utility Functions
# ---------------------------------------------------------------------------------------
def get_state_transition_matrix(ndim,dt):

    _dt = np.array(dt)

    if _dt.ndim != 0 and _dt.ndim != 1:
        raise ValueError, 'dt must be scalar or 1-d array'

    if _dt.ndim == 0:

        A = np.eye(3*ndim)
        for i in range(2*ndim):
            A[i,i+ndim] = _dt
        for i in range(ndim):
            A[i,i+2*ndim] = 0.5*_dt**2
        return A
    else:
        A_array = np.zeros((_dt.shape[0], 3*ndim, 3*ndim))
        for i, dt_item in enumerate(_dt):
            A_array[i,:,:] = get_state_transition_matrix(ndim,dt_item)
        return A_array


def get_observation_matrix(ndim):
    C = np.zeros((ndim, 3*ndim))
    C[:ndim,:ndim] = np.eye(ndim)
    return C

def get_state_transition_covariance_matrix(ndim,dt,qval):
    _dt = np.array(dt)
    if _dt.ndim !=0 and _dt.ndim != 1:
        raise ValueError, 'dt must be scalar or 1-d array'

    if _dt.ndim ==1:
        dt_mean = _dt.mean()
    else:
        dt_mean = dt

    Q = np.eye(3*ndim)
    for i in range(ndim):
        Q[i,i] = 1.0 
        Q[i+ndim, i+ndim] = 1.0/dt_mean**2
        Q[i+2*ndim, i+2*ndim] = 1.0/(dt_mean**4)
    Q = qval*Q
    return Q

def get_observation_covariance_matrix(ndim,rval):
    R = rval*np.eye(ndim)
    return R

def get_initial_state_mean(x0):
    return np.array([x0[0], x0[1], 0.0, 0.0, 0.0, 0.0])

def get_initial_state_covariance_matrix(ndim):
    return np.eye(3*ndim)

