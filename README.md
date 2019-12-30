# time-bound-computation
Matrix multiplication is used for producing a single matrix from two matrices. It's computation has a general time complexity of O(N^3). This project shows its time bound computation using numpy, pure python, numba and a C based shared library.

## Prerequisites
python3  
numba module for python3

## Computation Graphs
![Graph 1](https://raw.githubusercontent.com/r4j0x00/time-bound-computation/master/images/mmat1.png?token=AHPFB5TMX3CWAJJ2Z5M23YK6CM72W)
![Graph 2](https://raw.githubusercontent.com/r4j0x00/time-bound-computation/master/images/mmat2.png?token=AHPFB5WMI7JPILPR2G2RP326CNABI)

## Description
Looking at the graphs we notice numpy is much more faster than our pure python code. This is because numpy compiles the loops before executing them. Compiled code is much faster than the python interpreter. We also see that in the shared library which is compiled as well. Adding @numba.jit to our python function also makes it almost as fast as numpy. numba is a just in time compiler that compiles parts of our python code and then executes it, resulting in fast execution.
