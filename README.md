## h5_logger 

A dead simple implementation of a Kalman filter with a constant acceleration model (based on pykalman).


## Installation

Requirements: pykalman, numpy, matplotlib 

```bash
$ python setup.py install 

```


## Example

``` python
import numpy as np
import matplotlib.pyplot as plt
from const_accel_kalman import ConstAccelKalmanFilter

data = np.loadtxt('data.txt')

dt = 1/60.0 
qval = 0.02
rval = 1.0
t = np.arange(data.shape[0])*dt

kf = ConstAccelKalmanFilter(dt, qval, rval) 
data_filt, cov_filt = kf.smooth(data)

```




