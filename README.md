## const_accel_kalman 

A dead simple implementation of a Kalman filter with a constant acceleration model (based on pykalman).


## Installation

Requirements: pykalman, numpy, matplotlib 

```bash
$ python setup.py install 

```


## Example w/ constant dt

``` python
import numpy as np
from const_accel_kalman import ConstAccelKalmanFilter

data = np.loadtxt('data.txt')
dt = 1/60.0 
qval = 0.02
rval = 1.0

kf = ConstAccelKalmanFilter(qval, rval) 
data_filt, cov_filt = kf.smooth(data, dt)

```

## Example w/ variable dt


``` python
import numpy as np
from const_accel_kalman import ConstAccelKalmanFilter

data = np.loadtxt('data.txt')
dt = np.loadtxt('dt.txt')
qval = 0.02
rval = 1.0

kf = ConstAccelKalmanFilter(qval, rval) 
data_filt, cov_filt = kf.smooth(data, dt)

```


