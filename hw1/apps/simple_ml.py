from matplotlib.pyplot import axes
from needle.ops import exp, mean, log, summation, reshape, multiply, relu
import struct
import gzip
import numpy as np

import sys

from torch import Tensor
sys.path.append('python/')
import needle as ndl

def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded 
                data.  The dimensionality of the data should be 
                (num_examples x input_dim) where 'input_dim' is the full 
                dimension of the data, e.g., since MNIST images are 28x28, it 
                will be 784.  Values should be of type np.float32, and the data 
                should be normalized to have a minimum value of 0.0 and a 
                maximum value of 1.0. The normalization should be applied uniformly
                across the whole dataset, _not_ individual images.

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """

    # extract images
    images = []
    with gzip.open(image_filename, 'rb') as fp:
        header = struct.unpack('>4i', fp.read(16))
        magic, size, width, height = header

        if magic != 2051:
            raise RuntimeError("'%s' is not an MNIST image set." % f)

        chunk = width * height
        for _ in range(size):
            img = struct.unpack('>%dB' % chunk, fp.read(chunk))
            img_np = np.array(img, np.float32)
            images.append(img_np/255)

    # extract labels
    with gzip.open(label_filename, 'rb') as fp:
        header = struct.unpack('>2i', fp.read(8))
        magic, size = header

        if magic != 2049:
            raise RuntimeError("'%s' is not an MNIST label set." % f)

        labels = struct.unpack('>%dB' % size, fp.read())

    return (np.array(images), np.array(labels, np.uint8))


def softmax_loss(Z, y_one_hot):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION
    Zy = reshape(summation(multiply(Z, y_one_hot), axes = (1,)), (Z.shape[0], 1))
    Zf = log(reshape(summation(exp(Z), axes=(1,)), (Z.shape[0], 1)))
    return summation(Zf- Zy) / Z.shape[0]
    #return mean(Zf - Zy)
    ### END YOUR SOLUTION


def nn_epoch(X, y, W1, W2, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ### BEGIN YOUR SOLUTION
    for i in range(0, y.shape[0], batch):
        X_batch = ndl.Tensor(X[i:i + batch])
        y_batch = y[i:i + batch]

        Z = relu(X_batch @ W1) @ W2

        y_one_hot = np.zeros((y_batch.shape[0], Z.shape[-1]))
        y_one_hot[np.arange(y_batch.size), y_batch] = 1
        y_ = ndl.Tensor(y_one_hot)

        loss = softmax_loss(Z,y_)
        loss.backward()

        W1_grad = (W1.grad).numpy()
        W2_grad = (W2.grad).numpy()
        W1 = W1.numpy()
        W2 = W2.numpy()
        
        W1 = ndl.Tensor(W1 + (- lr * W1_grad))
        W2 = ndl.Tensor(W2 + (- lr * W2_grad))

    return (W1, W2)
    ### END YOUR SOLUTION


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h,y):
    """ Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h,y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
