"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []




class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(self, in_features, out_features, add_bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, 
                                                    requires_grad=True, device=device, dtype=dtype))
        if add_bias == True:
            self.bias = Parameter(init.kaiming_uniform(out_features, 1,
                                     requires_grad=True, device=device, dtype=dtype).reshape((1, out_features)))
        else:
            self.bias = None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        X_mul_weight = X @ self.weight
        if self.bias is not None:
            return X_mul_weight + self.bias.broadcast_to(X_mul_weight.shape)
        else:
            return X_mul_weight
        ### END YOUR SOLUTION



class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        dim = X.shape
        flat = 1
        for d in dim[1:]:
            flat *= d
        return ops.reshape(X, (dim[0], flat))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
            x = module(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        m, n = logits.shape
        y_hat = ops.summation((logits * init.one_hot(n, y)), (1,))
        return ops.summation(ops.logsumexp(logits, (1,)) - y_hat) / m
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, requires_grad=True, dtype=dtype, device=device))
        self.bias = Parameter(init.zeros(dim, requires_grad=True, dtype=dtype, device=device))
        self.running_mean = init.zeros(dim, dtype=dtype, device=device)
        self.running_var = init.ones(dim, dtype=dtype, device=device)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch, dim = x.shape
        if self.training == False:
            x_hat = (x - (self.running_mean).broadcast_to(x.shape)) / \
                    ((self.running_var + self.eps)**0.5).broadcast_to(x.shape)
        else:
            Ex = ops.summation(x, (0,)) / batch

            Vx = ops.summation((x - Ex.broadcast_to(x.shape))**2, (0,)) / batch

            x_hat = (x - Ex.broadcast_to(x.shape)) / \
                ((Vx + self.eps)**0.5).broadcast_to(x.shape)

            # Update the mean and variance using moving average
            self.running_mean = (1.0 - self.momentum) * self.running_mean + self.momentum * Ex.data
            self.running_var = (
                1.0 - self.momentum) * self.running_var + self.momentum * Vx.data

        return (self.weight).broadcast_to(x.shape) * x_hat + (self.bias).broadcast_to(x.shape)
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, requires_grad=True, dtype=dtype, device=device))
        self.bias = Parameter(init.zeros(dim, requires_grad=True, dtype=dtype, device=device))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch, dim = x.shape
        Ex = ops.reshape(ops.summation(x, (1,)) / self.dim, (batch, 1)).broadcast_to(x.shape)
        Vx = ops.reshape(ops.summation(ops.power_scalar(
            x - Ex, 2), (1,)) / dim, (batch, 1)).broadcast_to(x.shape)

        return (self.weight).broadcast_to(x.shape) * ((x - Ex) / ops.power_scalar((Vx + self.eps), 0.5)) + (self.bias).broadcast_to(x.shape)
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training == True:
            prob = init.randb(*(x.shape), p = 1 - self.p)
            return (x * prob) / (1 - self.p)
        else:
            return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION