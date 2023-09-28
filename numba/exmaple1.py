import numpy as np
import time

from numba import guvectorize

def move_mean_normal(a, window_arr):
    out=np.empty(a.size, dtype=object)
    a = a.flatten()
    window_width = window_arr
    asum = 0.0
    count = 0
    for i in range(window_width):
        asum += a[i]
        count += 1
        out[i] = asum / count
    for i in range(window_width, len(a)):
        asum += a[i] - a[i - window_width]
        out[i] = asum / count
    return out

@guvectorize(['void(float64[:], intp[:], float64[:])'],
             '(n),()->(n)')
def move_mean_numba(a, window_arr, out):
    window_width = window_arr[0]
    asum = 0.0
    count = 0
    for i in range(window_width):
        asum += a[i]
        count += 1
        out[i] = asum / count
    for i in range(window_width, len(a)):
        asum += a[i] - a[i - window_width]
        out[i] = asum / count

row = 1000
col = 1000
N = row * col
arr = np.arange(N, dtype=np.float64).reshape(row, col)
print(arr)
print("=============================")
start = time.time()
print(move_mean_normal(arr, 10))
print(time.time()-start)
print("=============================")
start = time.time()
print(move_mean_numba(arr, 10))
print(time.time()-start)

