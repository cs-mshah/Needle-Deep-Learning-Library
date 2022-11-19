import numpy as array_api

import sys
sys.path.append('../python/')
import needle as ndl
from needle.ops import *
from needle.init import *
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

y = ndl.Tensor([1,2,4])

a = randn(3,11)
print(one_hot(5,y))
