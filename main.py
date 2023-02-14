import torch
import torch.nn as nn
import numpy as np


class train():
    pass


class args():
    def __init__(self, batch_size, num_inputs, num_outputs, num_hiddens, learning_rate, num_epochs):
        self.batch_size = batch_size
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_hiddens = num_hiddens
        self.lr = learning_rate
        self.num_epochs = num_epochs

    pass


class Dataloader():
    pass


class Model(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens):
        self.W1 = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_hiddens)), dtype=torch.float)
        self.B1 = torch.zeros(num_hiddens, dtype=torch.float)
        self.W2 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens, num_outputs)), dtype=torch.float)
        self.B2 = torch.zeros(num_outputs, dtype=torch.float)

    def relu(self, X):
        return torch.max(input=X, other=torch.tensor(0.0))

    def loss(self):
        return torch.nn.CrossEntropyLoss()

    def net(self, X, num_inputs):
        X = X.view((-1, num_inputs))
        H = self.relu(torch.matmul(X, self.W1) + self.B1)
        return torch.matmul(H, self.W2) + self.B1
