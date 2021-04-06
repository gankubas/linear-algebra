# Matrix template for optimizing CROUT factorization
# x 0 0 0 ... 0 0 X
# X X 0 0 ... 0 0 X
# X 0 X 0 ... 0 0 X
# X 0 0 X ... 0 0 X
# .................
# X 0 0 0 ... X 0 X
# X 0 0 0 ... 0 X X
# X 0 0 0 ... 0 0 X

import numpy as np

rng = np.random.default_rng(0)

#------------ Input --------------------------------
n = 5;
A = np.diagflat(rng.integers(1,10, size=(n,1)), k=0)
A[:, n-1] = rng.integers(1,20, size=(1,n))
A[:, 0] = rng.integers(1,20, size=(1,n))
A = A.astype(float);
b = rng.integers(1,10, size=(n,1)).astype(float)
#-----------------------------------------------------

print('-------------A------------------')
print(A)
print('-----------b--------------------')
print(b)

sol = np.linalg.inv(np.copy(A))@np.copy(b)

def Crout(A):
    n = len(A)
    U = np.eye(n)
    L = np.zeros((n, n))

    for i in range(n):
        L[i][0] = A[i][0]
    U[0][n - 1] = A[0][n - 1] / L[0][0]

    for k in range(1, n):
        L[k][k] = A[k][k]
        U[k][n - 1] =(A[k][n - 1] - L[k][0] * U[0][n - 1]) / L[k][k]

    return L, U

def LTRIS(L, b):
    n = len(L)
    y = np.zeros((n, 1))

    y[0] = b[0] / L[0][0]
    for i in range(1, n):
        s = b[i]
        s = s - L[i][0] * y[0] - L[i][i] * y[i]
        y[i] = s / L[i][i]
    return y

def UTRIS(U, y):
    n = len(U)
    x = np.zeros((n, 1))

    x[n - 1] = y[n - 1] / U[n - 1][n - 1]
    for i in range(n - 2, -1, -1):
        s = y[i]
        s = s - U[i][n - 1] * x[n - 1] - x[i]
        x[i] = s / U[i, i]
    return x

[L, U] = Crout(np.copy(A))
print('----------- L ---------------')
print(L)
print('----------- U ---------------')
print(U)

print('----------- A_obt---------------')
print(L @ U)

y = LTRIS(L, b)
x = UTRIS(U, y)

print('----------x------------')
print(x)
print('------------sol----------')
print(sol)
