"""Train the Homework 3 CNN on CPU for one epoch and save its weights.

Training/evaluation adapted from https://github.com/pytorch/examples/tree/main/mnist
Copyright (c) 2017, Pytorch contributors. BSD 3-Clause; see LICENSE.
"""

import os
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from model import Net, transform


def main():
    data_dir = os.environ.get("DATA_DIR", "data")
    model_path = Path(os.environ.get("MODEL_PATH", "model/mnist_cnn.pt"))
    train_data = MNIST(data_dir, train=True, transform=transform, download=False)
    test_data = MNIST(data_dir, train=False, transform=transform, download=False)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=256)

    model = Net().to("cpu")
    optimizer = torch.optim.Adadelta(model.parameters(), lr=1.0)
    model.train()
    for batch_index, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        loss = F.nll_loss(model(images), labels)
        loss.backward()
        optimizer.step()
        if batch_index % 100 == 0:
            print(f"Epoch 1, batch {batch_index}/{len(train_loader)}, loss {loss.item():.4f}",
                  flush=True)

    model.eval()
    correct = 0
    with torch.inference_mode():
        for images, labels in test_loader:
            predicted = model(images).argmax(dim=1)
            correct += (predicted == labels).sum().item()
    print(f"Test accuracy: {correct}/{len(test_data)} "
          f"({100 * correct / len(test_data):.2f}%)", flush=True)

    # On Kubernetes, /model is the persistent volume shared with inference.
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}", flush=True)


if __name__ == "__main__":
    main()
