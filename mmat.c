#include <stdio.h>
#include <stdlib.h>

// gcc -shared -fPIC mmat.c -o mmat.so

long long **mmat(int n, long long array[n][n], long long array2[n][n]) {
	long long **c = (long long**)malloc(n * sizeof(long long*)); // using dynamic memory to not lose data after the function returns
	for (int i=0;i<n;++i) c[i] = (long long*)malloc(n*sizeof(long long));
	for (int i=0;i<n;++i) {
		for (int j=0;j<n;++j) {
			long long s = 0;
			for(int x=0;x<n;++x) {
				s += array[i][x] * array2[x][j];
			}
			c[i][j] = s;
		}
	}
	return c;
}
