# -*- coding: utf-8 -*-
import argparse
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from model import cifarNet, LeNet, weights_init


def label_to_onehot(target, num_classes=100):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target

def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))


device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
print("Running on %s" % device)

def dlg_attack(model, dataset, epoch, idx):
    if dataset == "mnist":
        dst = datasets.MNIST("./data/mnist", download=True, train=True)
    elif dataset == "fmnist":
        dst = datasets.FashionMNIST("./data/FMNIST", download=True, train=True)
    elif dataset == "cifar10":
        dst = datasets.CIFAR10("./data/CIFAR10", download=True)
    else:
        raise ValueError("No such dataset!")

    tp = transforms.ToTensor()
    tt = transforms.ToPILImage()

    img_index = idx
    gt_data = tp(dst[img_index][0]).to(device)

    gt_data = gt_data.view(1, *gt_data.size())
    gt_label = torch.Tensor([dst[img_index][1]]).long().to(device)
    gt_label = gt_label.view(1, )
    gt_onehot_label = label_to_onehot(gt_label, num_classes=10)

    # plt.imshow(tt(gt_data[0].cpu()), cmap="gray")
    plt.imshow(tt(gt_data[0].cpu()))
    plt.show()
    img = tt(gt_data[0].cpu())
    img.save(f"./img/{dataset}/gt_data_idx{img_index}.png")
    # plt.imsave(f"./img/{dataset}/gt_data_idx{img_index}.png", tt(gt_data[0].cpu()))

    net = model.to(device)

    # torch.manual_seed(1234)

    criterion = cross_entropy_for_onehot

    # compute original gradient
    pred = net(gt_data)
    y = criterion(pred, gt_onehot_label)
    dy_dx = torch.autograd.grad(y, net.parameters())

    original_dy_dx = list((_.detach().clone() for _ in dy_dx))

    # generate dummy data and label
    dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
    dummy_label = torch.randn(gt_onehot_label.size()).to(device).requires_grad_(True)

    plt.imshow(tt(dummy_data[0].cpu()))

    optimizer = torch.optim.LBFGS([dummy_data, dummy_label])

    history = []
    for iters in range(300):
        def closure():
            optimizer.zero_grad()

            dummy_pred = net(dummy_data)
            dummy_onehot_label = F.softmax(dummy_label, dim=-1)
            dummy_loss = criterion(dummy_pred, dummy_onehot_label)
            dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)

            grad_diff = 0
            for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                grad_diff += ((gx - gy) ** 2).sum()
            grad_diff.backward()

            return grad_diff

        optimizer.step(closure)
        if iters % 10 == 0:
            current_loss = closure()
            print(iters, "%.4f" % current_loss.item())
            history.append(tt(dummy_data[0].cpu()))
            img = tt(dummy_data[0].cpu())
            img.save(f"./img/{dataset}/dummy_data_idx{img_index}_epoch{iters}.png")
            # plt.imsave(f"./img/{dataset}/dummy_data_idx{img_index}_epoch{iters}.png", tt(dummy_data[0].cpu()))

    plt.figure(figsize=(12, 8))
    for i in range(30):
        plt.subplot(3, 10, i + 1)
        plt.imshow(history[i])
        plt.title("iter=%d" % (i * 10))
        plt.axis('off')

    plt.show()

# model = LeNet().to(device)
# model.apply(weights_init)
# dlg_attack(model=model, dataset='cifar10', epoch=0, idx=1)