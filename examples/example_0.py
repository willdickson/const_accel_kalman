from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from const_accel_kalman import ConstAccelKalmanFilter

data = np.loadtxt('data.txt')

dt = 1/60.0 
qval = 0.002
rval = 1.0
t = np.arange(data.shape[0])*dt

kf = ConstAccelKalmanFilter(qval, rval) 
data_filt, cov_filt = kf.smooth(data,dt)

x = data[:,0]
y = data[:,1]
x_filt = data_filt[:,0]
y_filt = data_filt[:,1]
dx_filt = data_filt[:,2]
dy_filt = data_filt[:,3]
speed = np.sqrt( dx_filt**2 + dy_filt**2)

plt.figure(1)
x = data[:,0]
y = data[:,1]
plt.plot(x,y,'b')
plt.plot(x_filt,y_filt,'r')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.plot()

plt.figure(2)
ax = plt.subplot(211)
plt.plot(t,x,'b')
plt.plot(t,x_filt,'r')
plt.ylabel('x')
plt.subplot(212,sharex=ax)
plt.plot(t,dx_filt,'r')
plt.xlabel('t')
plt.ylabel('dx')

plt.figure(3)
ax = plt.subplot(211)
plt.plot(t,y,'b')
plt.plot(t,y_filt,'r')
plt.ylabel('y')
plt.subplot(212,sharex=ax)
plt.plot(t,dy_filt,'r')
plt.xlabel('t')
plt.ylabel('dy')

plt.figure(4)
plt.plot(t,speed)
plt.xlabel('t')
plt.ylabel('speed')
plt.show()

