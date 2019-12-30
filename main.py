#!/usr/bin/env python3
from random import shuffle, randrange
from numpy import array, dot, zeros, shape, random
import time
from ctypes import cdll, c_longlong, c_int, POINTER
from numba import jit,njit
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
lib = cdll.LoadLibrary('bin/mmat.so')

def multiply_matrix_cdll(a,b):
    assert len(a[0]) == len(a) == len(b[0]) == len(b)
    n = len(a)
    seq = (c_longlong*n)*n
    arr = seq()
    arr2 = seq()

    for i in range(n):
        for j in range(n):
            arr[i][j] = c_longlong(a[i][j])
            arr2[i][j] = c_longlong(b[i][j])

    lib.mmat.restype = POINTER(POINTER(c_longlong*n)*n)
    res = lib.mmat(n,arr,arr2)
    f = []
    for i in res.contents:
        f.append(list(i.contents))
    return f

def multiply_matrix(a,b):
    assert len(a[0]) == len(a) == len(b[0]) == len(b)
    f = []
    n = len(a)
    for i in range(n):
        r = []
        for j in range(n):
            c = 0
            for x in range(n):
                c += a[i][x]*b[x][j]
            r.append(c)
        f.append(r)
    return f


@jit('void(int64[:,:],int64[:,:],int64[:,:])')
def multiply_matrix_jit(a,b,c):
    assert len(a[0]) == len(a) == len(b[0]) == len(b)
    n = len(a)
    for i in range(n):
        for j in range(n):
            for x in range(n):
                c[i][j] += a[i][x] * b[x][j]

def get_func_time(func,*args):
    start = time.process_time()
    func(*args)
    return (time.process_time()-start)

#r = range(10,500,10)
r = range(1,200)
plt.title('Matrix multiplication')
plt.ylabel('Execution time')
plt.xlabel('Size')

times = [[][::] for i in range(4)]

for i in r:
    m1 = random.randint(10000,size=(i,i))
    m2 = random.randint(10000,size=(i,i))
    times[0].append(get_func_time(multiply_matrix_cdll, m1, m2))
    times[1].append(get_func_time(multiply_matrix, m1, m2))
    times[2].append(get_func_time(dot, m1, m2))
    c = zeros(shape=(i,i),dtype='int64')
    times[3].append(get_func_time(multiply_matrix_jit, m1,m2,c))

plt.plot(r,times[0], color='blue',label='cdll shared library')
plt.plot(r,times[1], color='red',label='pure python')
plt.plot(r,times[2], color='green',label='numpy')
plt.plot(r,times[3], color='brown',label='numba jit')

plt.legend()
plt.show()
