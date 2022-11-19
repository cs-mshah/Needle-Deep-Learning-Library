import numpy as array_api

import sys
sys.path.append('../python/')
import needle as ndl
import needle.ops
from collections import defaultdict
from typing import Dict, List, DefaultDict

# print(ndl.divide(ndl.Tensor([[3.4, 2.35, 1.25], [0.45, 1.95, 2.55]]),
#                  ndl.Tensor([[4.9, 4.35, 4.1], [0.65, 0.7, 4.04]])).numpy())

arr = array_api.array([
    [
        [1,2], 
        [4,5], 
        [5,6]
    ], 
    [
        [3,1], 
        [2,7], 
        [9,8]
    ]
    ])

print(arr, arr.shape)
# print(arr.sum(axis=(0,2)).shape)


# def test_func(dici: DefaultDict[str, List[int]]):
#     l = ['a', 'b', 'a', 'd']
#     cnt = 0
#     for i in l:
#         dici[i].append(cnt)
#         cnt += 1

# di:DefaultDict[str, List[int]] = defaultdict(list)
# test_func(di)
# print(di)


# a = array_api.array([7, 1, 5])
# broad = array_api.array([5, 8, 7, 6, 5])

# axes = []
# axes.extend(range(len(broad) - len(a)))

# position = len(broad) - len(a)
# sum_axes = list(range(position))
# for i in range(position, len(broad)):
#     if a[i - position] != broad[i]:
#         sum_axes.append(i)

# print(sum_axes)

a = array_api.array([7, 1, 5, 6])

c = array_api.max(arr, axis=(0, 2))
print(c, c.shape)

T = ndl.Tensor(arr)
print(len(T))