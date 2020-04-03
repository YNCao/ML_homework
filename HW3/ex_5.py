# -*- coding: utf-8 -*-
import time
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

# load data and convert to full matrix
data = sio.loadmat('./hw3.mat')
y = data['y']
X = data['X']
X = np.array(X.todense())
n, d = X.shape
d_ = d + 1
X_ = np.column_stack((np.ones((n,1)), X))


# 1. normalize X by column vector
Z = (X-X.min(0)) / (X.max(0)-X.min(0))
Z_ = np.column_stack((np.ones((n,1)), Z))
del data, X, Z


# 2. Lipschitz consant 
print('calculating L_f...')
xtx = np.matmul(X_.T, X_)
L_f = 2/n*np.linalg.norm(xtx, ord=2)
print('L_f=', L_f)
print('calculating L_g...')
ztz = np.matmul(Z_.T, Z_)
L_g = 2/n*np.linalg.norm(ztz, ord=2)
print('L_g=', L_g)


# 3. closed form for g(u)
print('calculating closed form solution...')
t0 = time.time()
u1 = np.matmul(np.linalg.inv(ztz),Z_.T)
u = np.matmul(u1,y)
g = 1/n*(np.linalg.norm(y-np.matmul(Z_,u), ord=2))**2
t1 = time.time()
print('time consumption: ', t1-t0, 's\n optimal value g*: ', g)
 

# 4. gradient descent
alpha = 0.5
err = 0.01
i = 0
uu = [np.zeros((d_, 1))]
gg = [1/n*(np.linalg.norm(y-np.matmul(Z_, uu[0]), ord=2))**2]
yy = np.matmul(Z_.T, y)

tt0=time.time()
while abs(gg[i]-g)>err:
    uu.append(uu[i]-alpha*2/n*(np.matmul(ztz, uu[i]) - yy))
    gg.append(1/n*(np.linalg.norm(y-np.matmul(Z_, uu[i+1]), ord=2))**2)
    i += 1
    print('times: ', i,'||measure: ',gg[i],'||err: ',abs(gg[i]-g))
tt1=time.time()

Z_ = X_