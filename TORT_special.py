# Matrix template for optimizing orthogonal triangulation
# x x x x ... x X X
# 0 X X X ... X X X
# 0 0 X X ... X X X
# 0 0 0 X ... X X X
# .................
# 0 0 0 0 ... X X X
# 0 0 0 0 ... 0 X X
# 0 0 0 0 ... 0 0 X
# 0 0 0 0 ... 0 0 0
# X X X X ... X X X
# X X X X ... X X X

import numpy as np
from math import sqrt

rng = np.random.default_rng(0)

# ----------- Input ---------------------------
m = 7
n = 4
A = rng.integers(1,11, size=(m,n))
A = np.triu(A)
A[m-2,:] = rng.integers(1,10, size=(1,n))
A[m-1,:] = rng.integers(1,10, size=(1,n))
A = A.astype(float)
b = rng.integers(10, size=(m,1)).astype(float)
#-----------------------------------------------
print('------------A-------------\n', A)

sol = np.linalg.pinv(np.copy(A))@np.copy(b)

def TORT(A):
    m, n = np.shape(A)
    U = np.zeros((m, n))

    p = min(m,n)
    beta = np.zeros(p)

    for k in range(p):
        sigma = A[k][k] * A[k][k] + A[m - 2][k] * A[m - 2][k] + A[m - 1][k] * A[m - 1][k]
        sigma = np.sign(A[k][k]) * sqrt(sigma)

        if(sigma == 0 ):
            beta[k] = 0
        else:
            U[k][k] = A[k][k] + sigma
            U[m - 2][k] = A[m - 2][k]
            U[m - 1][k] = A[m - 1][k]

            beta[k] = sigma * U[k][k]

            A[k][k] = 0 - sigma
            A[m - 2][k] = 0
            A[m - 1][k] = 0

            for j in range(k + 1, n):
                tau = U[k][k] * A[k][j] + U[m - 2][k] * A[m - 2][j] + U[m - 1][k] * A[m - 1][j]
                tau /= beta[k]

                A[k][j] -= tau * U[k][k]
                A[m - 2][j] -= tau * U[m - 2][k]
                A[m - 1][j] -= tau * U[m - 1][k]

    return U, A[:n, :], beta

def CMMP(b, U, beta):
    m, n = np.shape(U)

    for k in range(n):
        tau = U[k][k] * b[k] + U[m - 2][k] * b[m - 2] + U[m - 1][k] * b[m - 1]
        tau /= beta[k]

        b[k] -= tau * U[k][k]
        b[m - 2] -= tau * U[m - 2][k]
        b[m - 1] -= tau * U[m - 1][k]

    return b

def Utris(U, b):
    n = len(U)
    x = np.zeros((n,1))
    for i in range(n-1,-1, -1):
        s = b[i]
        for j in range(i+1,n):
            s = s -U[i][j]*x[j]
        x[i] = s/U[i][i]
    return x

U, R, beta  = TORT(A)

print('----------U------------\n', U)
print('-----------R-----------\n',R)

b = CMMP(b, U, beta)
x = Utris(R, b)

# Verificare
print('-----------x------------\n',x)
print('----------sol------------\n', sol)
