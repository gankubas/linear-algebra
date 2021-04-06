# Matrix template for optimizing gaussian elimination with partial pivoting
# x x x x ... x X X
# 0 X X X ... X X X
# 0 0 X X ... X X X
# 0 0 0 X ... X X X
# .................
# 0 0 0 0 ... X X X
# 0 0 0 0 ... 0 X X
# X X X X ... X X X

import numpy as np
import time
import math as m

rng = np.random.default_rng(0)

#----------- Input --------------
n = 5
A = rng.integers(-10,11, size = (n,n)) +1
A = np.triu(A)
A[n-1,:] = rng.integers(-10,11, size = (1,n)) +1
b =  rng.integers(-10,11, size = (n,1)).astype('float')
A = A.astype('float')
#-------------------------------------
print('---------- A -----------')
print(A)
print('---------- b -----------')
print(b)

def EGPP(A, b):

    maxx=np.zeros((n,n))
    for k in range(n-1):
        if(m.fabs(A[k][k])<m.fabs(A[n-1][k])):
            A[[n-1,k]] = A[[k,n-1]]
            b[[n - 1, k]] = b[[k, n - 1]]

        maxx[n-1][k]=A[n-1][k]/A[k][k]
        for j in range(k,n,1):
            for i in range(k+1,n,1):
                A[i][j]=A[i][j]-maxx[i][k]*A[k][j]
                b[i][0]=b[i][0]-maxx[i][k]*b[k][0]
    return A, b


def Utris(U, b,n):
    x = np.empty(n)
    for i in range(n - 1, -1, -1):
        s = b[i]
        for j in range(n - 1, i, -1):
            s = s - U[i, j] * x[j]
        x[i] = s / U[i, i]
    return x

[U,bb] = EGPP(A,b)
x = Utris(U,bb, n)
print('----------x------------')
print(x)

sol = np.linalg.inv(A)@b
print('------------sol----------')
print(sol)
