import numpy as np

# 2D array 
arr = np.array([[14, 17, 12, 33, 44],  
       [15, 6, 27, 8, 19], 
       [23, 2, 54, 1, 4,]])


ind = np.array([2, 0, 1])
print(np.take(arr, ind, axis=1))
print(arr[np.arange(arr.shape[0]), ind])

a1 = [[1,2], [2,4]]
a2 = [[2,1], [12,1]]
print(np.multiply(a1, a2))

# a = np.array([1,2,3,4,5])
# print(a[2:15])

arr = np.zeros((3,5))

print(arr)

ind = np.array([2, 0, 1])

arr[np.arange(ind.size), ind] = 1

print(arr)

print(arr[: , ind])

