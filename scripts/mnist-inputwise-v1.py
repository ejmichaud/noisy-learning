#!/usr/bin/env python
# coding: utf-8

from itertools import islice
import random

import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torchvision

from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
ex = Experiment("mnist-batch-noise-v1")
ex.captured_out_filter = apply_backspaces_and_linefeeds

class NoisyLinear(nn.Linear):
    def __init__(self, *args, scale, device, **kwargs):
        super(NoisyLinear, self).__init__(*args, **kwargs)
        self.scale = scale
        self.device = device
    
    def forward(self, x):
        b = x.shape[-2]
        weight_tiled = torch.tile(self.weight, (b, 1, 1))
        weight_noise = self.scale * torch.randn(weight_tiled.shape).to(self.device)
        weight_noisy = weight_tiled + weight_noise
        if self.bias is not None:
            bias_tiled = torch.tile(self.bias, (b, 1))
            bias_noise = self.scale * torch.randn(bias_tiled.shape).to(self.device)
            bias_noisy = bias_tiled + bias_noise
        product = torch.matmul(x, weight_noisy.transpose(2, 1))
        product = torch.diagonal(product, dim1=-3, dim2=-2)
        if self.bias is not None:
            return product.T + bias_noisy
        return product.T


def compute_accuracy(network, dataset, device, N=2000, batch_size=50):
    with torch.no_grad():
        dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        correct = 0
        total = 0
        for x, labels in islice(dataset_loader, N // batch_size):
            logits = network(x.to(device))
            predicted_labels = torch.argmax(logits, dim=1)
            correct += torch.sum(predicted_labels == labels.to(device))
            total += x.size(0)
        return (correct / total).item()

def compute_loss(network, loss_fn, dataset, device, N=2000, batch_size=50):
    with torch.no_grad():
        dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        loss = 0
        total = 0
        for x, labels in islice(dataset_loader, N // batch_size):
            logits = network(x.to(device))
            loss += loss_fn(logits, labels.to(device))
            total += x.size(0)
        return (loss / total).item()


@ex.config
def cfg():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float64
    epochs = 100
    scales = [0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.07, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.25, 0.3, 0.5]


@ex.automain
def run(scales, epochs, device, dtype, seed):
    torch.set_default_dtype(dtype)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    train = torchvision.datasets.MNIST(root="/tmp", train=True, transform=torchvision.transforms.ToTensor(), download=True)
    test = torchvision.datasets.MNIST(root="/tmp", train=False, transform=torchvision.transforms.ToTensor(), download=True)
    train_loader = torch.utils.data.DataLoader(train, batch_size=50, shuffle=True)

    for scale in tqdm(scales):
        ex.info[scale] = dict() # results
        
        network = nn.Sequential(
            nn.Flatten(),
            NoisyLinear(28*28, 300, scale=scale, device=device),
            nn.ReLU(),
            NoisyLinear(300, 100, scale=scale, device=device),
            nn.ReLU(),
            NoisyLinear(100, 10, scale=scale, device=device),
        ).to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(network.parameters())
        network.train()

        ex.info[scale]['batches'] = list()
        ex.info[scale]['train_losses'] = list()
        ex.info[scale]['test_losses'] = list()
        ex.info[scale]['train_accuracies'] = list()
        ex.info[scale]['test_accuracies'] = list()
        
        batch_n = 0
        for epoch in tqdm(range(epochs), leave=False):
            network.train()
            for x, label in train_loader:
                if batch_n % 200 == 0:
                    network.eval()
                    with torch.no_grad():
                        ex.info[scale]['batches'].append(batch_n)
                        ex.info[scale]['train_losses'].append(compute_loss(network, loss_fn, train, device))
                        ex.info[scale]['test_losses'].append(compute_loss(network, loss_fn, test, device))
                        ex.info[scale]['train_accuracies'].append(compute_accuracy(network, train, device))
                        ex.info[scale]['test_accuracies'].append(compute_accuracy(network, test, device))
                    network.train()
                optimizer.zero_grad()
                loss = loss_fn(network(x.to(device)), label.to(device))
                loss.backward()
                optimizer.step()
                batch_n += 1
        
