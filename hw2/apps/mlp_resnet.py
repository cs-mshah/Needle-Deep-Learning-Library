import sys
sys.path.append('../python')
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)

def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION

    # create a tuple of moduels
    # note the norm parameter. directly callable
    modules = nn.Sequential(
        nn.Linear(dim, hidden_dim),
        norm(hidden_dim),
        nn.ReLU(),
        nn.Dropout(drop_prob),
        nn.Linear(hidden_dim, dim),
        norm(dim)
    )

    # add ReLU after Residual
    return nn.Sequential(
        nn.Residual(modules),
        nn.ReLU()
    )
    ### END YOUR SOLUTION


def MLPResNet(dim, hidden_dim=100, num_blocks=3, num_classes=10, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION

    # add Linear + ReLU
    modules = [nn.Sequential(
        nn.Flatten(),
        nn.Linear(dim, hidden_dim),
        nn.ReLU()
    )]

    # add num_blocks residual blocks with params
    for i in range(num_blocks):
        modules.append(ResidualBlock(
            hidden_dim, hidden_dim//2, norm, drop_prob))

    # add linear block after that
    modules.append(nn.Linear(hidden_dim, num_classes))

    # make a Sequential module from the modules, so unpack
    return nn.Sequential(*modules)
    ### END YOUR SOLUTION




def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION

    # instantiate loss function
    loss_func = nn.SoftmaxLoss()

    correct, loss_sum, n_step, n_samples = 0., 0., 0, 0

    # set mode of training
    if opt:
        model.train()
    else:
        model.eval()

    
    for X, y in dataloader:
        if opt:
            opt.reset_grad()
        pred = model(X)
        loss = loss_func(pred, y)
        correct += (pred.numpy().argmax(axis=1) == y.numpy()).sum()
        if opt:
            loss.backward()
            opt.step()
        loss_sum += loss.numpy()
        n_step += 1
        n_samples += X.shape[0]

    # NOTE (1 - mean of acc over iterations) is not accurate enough
    return (1 - correct / n_samples), loss_sum / n_step
    ### END YOUR SOLUTION



def train_mnist(batch_size=100, epochs=10, optimizer=ndl.optim.Adam,
                lr=0.001, weight_decay=0.001, hidden_dim=100, data_dir="data"):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    train_images_path = os.path.join(data_dir,'train-images-idx3-ubyte.gz')
    train_labels_path = os.path.join(data_dir, 'train-labels-idx1-ubyte.gz')
    test_images_path = os.path.join(data_dir, 't10k-images-idx3-ubyte.gz')
    test_labels_path = os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz')

    # load dataset
    train_data = ndl.data.MNISTDataset(train_images_path, train_labels_path)
    test_data = ndl.data.MNISTDataset(test_images_path, test_labels_path)

    # create dataloaders
    train_loader = ndl.data.DataLoader(train_data, batch_size)
    test_loader = ndl.data.DataLoader(test_data, batch_size)

    # instantiate model
    model = MLPResNet(28 * 28, hidden_dim)

    # set optimizer
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

    # train for epochs
    for _ in range(epochs):
        train_acc, train_loss = epoch(train_loader, model, opt)
        test_acc, test_loss = epoch(test_loader, model)

    return (train_acc, train_loss, test_acc, test_loss)
    ### END YOUR SOLUTION



if __name__ == "__main__":
    train_mnist(data_dir="../data")
