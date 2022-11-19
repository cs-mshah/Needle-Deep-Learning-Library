import numpy as np
from .autograd import Tensor
import struct
import gzip
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any


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
            images.append(img_np/255.0)

    # extract labels
    with gzip.open(label_filename, 'rb') as fp:
        header = struct.unpack('>2i', fp.read(8))
        magic, size = header

        if magic != 2049:
            raise RuntimeError("'%s' is not an MNIST label set." % f)

        labels = struct.unpack('>%dB' % size, fp.read())

    return (np.array(images), np.array(labels, np.uint8))


class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as an H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        if flip_img is True:
            # returns a view of the flipped image given the axis
            return np.flip(img, 1)
        else:
            return img
        ### END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return 
            H x W x C NDArray of clipped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(low=-self.padding, high=self.padding+1, size=2)
        ### BEGIN YOUR SOLUTION
        result = np.zeros_like(img)
        H, W = img.shape[0], img.shape[1]
        # when shift out of bounds, just return zeros
        if abs(shift_x) >= H or abs(shift_y) >= W:
            return result
        st_1, ed_1 = max(0, -shift_x), min(H - shift_x, H)
        st_2, ed_2 = max(0, -shift_y), min(W - shift_y, W)
        img_st_1, img_ed_1 = max(0, shift_x), min(H + shift_x, H)
        img_st_2, img_ed_2 = max(0, shift_y), min(W + shift_y, W)
        result[st_1:ed_1, st_2:ed_2, :] = img[img_st_1:img_ed_1, img_st_2:img_ed_2, :]
        return result
        ### END YOUR SOLUTION


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
     """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        indices = np.arange(len(dataset))
        if not self.shuffle:
            self.ordering = np.array_split(indices,
                                           range(batch_size, len(dataset), batch_size))
        else:
            np.random.shuffle(indices)
            self.ordering = np.array_split(indices,
                                           range(batch_size, len(dataset), batch_size))

    def __iter__(self):
        ### BEGIN YOUR SOLUTION
        self.start = 0
        ### END YOUR SOLUTION
        return self

    def __next__(self):
        ### BEGIN YOUR SOLUTION
        if self.start == len(self.ordering):
            raise StopIteration
        a = self.start
        self.start += 1
        samples = [Tensor(x) for x in self.dataset[self.ordering[a]]]
        return tuple(samples)
        ### END YOUR SOLUTION


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        super().__init__(transforms)
        images, self.labels = parse_mnist(image_filename, label_filename)
        self.images = images.reshape((images.shape[0], 28, 28, 1))
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        x, y = self.images[index], self.labels[index]
        x = self.apply_transforms(x)
        return x, y
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return len(self.images)
        ### END YOUR SOLUTION

class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])
