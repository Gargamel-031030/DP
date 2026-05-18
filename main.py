import os
import sys
import random
import time

import numpy as np
import torch
import copy
from torch.utils.data import DataLoader
from data import get_mnist_datasets, get_clients_datasets, get_fmnist_datasets, get_CIFAR10, get_noniid_fmnist
from dlg import dlg_attack
from model import *
from client import Client
from dpsgd_utils import *
from utils import *
from tqdm.auto import trange, tqdm
from options import parse_args
import torch.optim as optim
import pandas as pd

random.seed(10)

args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

num_clients = args.num_clients
local_epoch = args.local_epoch
global_epoch = args.global_epoch
batch_size = args.batch_size
user_sample_rate = args.user_sample_rate
dataset = args.dataset

target_epsilon = args.epsilon_file
target_delta = args.target_delta
clipping_bound = args.clipping_bound
alpha = args.alpha

fedavg = args.fedavg
weiavg = args.weiavg
deavg = args.deavg

nm_decay = args.nm_decay
decay_factor = args.decay_factor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.store:
    saved_stdout = sys.stdout
    # 构建目录路径
    dir_path = f'./txt/{target_epsilon}/'

    # 如果目录不存在，创建目录
    os.makedirs(dir_path, exist_ok=True)

    # 构建文件路径和文件名
    file_name = (
        f'{dir_path}'
        f'dataset_{dataset}_'
        f'num_clients_{num_clients}_'
        f'local_epoch_{local_epoch}_'
        f'global_epoch_{global_epoch}_'
        f'batch_size_{batch_size}_'
        f'target_delta_{target_delta}_'
        f'clipping_bound_{clipping_bound}_'
        f'fisher_threshold_{args.fisher_threshold}_'
        f'lambda_1_{args.lambda_1}_'
        f'lambda_2_{args.lambda_2}_'
        f'lr_{args.lr}_'
        f'alpha_{args.dir_alpha}.txt'
    )

    # 打开文件进行追加
    file = open(file_name, 'a')
    sys.stdout = file

def integrand(x, sigma):
    pi_tensor = torch.tensor(torch.pi, device=device)
    denominator = 1 / (torch.sqrt(2 * pi_tensor) * sigma) * torch.exp(-x**2 / (2 * sigma**2))
    numerator = 1 / (6 * sigma)
    inner_log = torch.log(numerator / denominator)
    outer_log = numerator * inner_log
    del pi_tensor, denominator, numerator, inner_log
    torch.cuda.empty_cache()
    return outer_log

def eps2level(client):
    client_eps = client.ba.epsilon
    if client_eps == 0.5:
        level = 1
    elif client_eps == 1.0:
        level = 2
    elif client_eps == 2.0:
        level = 3
    elif client_eps == 4.0:
        level = 4
    elif client_eps == 8.0:
        level = 5
    else:
        raise ValueError('Invalid client epsilon!')

    return level

def customloss(outputs, labels, type, param_diffs=None, sigma=None):
    ce_loss = F.cross_entropy(outputs, labels)
    if type == "R1":
        reg_loss = torch.sum(torch.stack([torch.norm(diff) for diff in param_diffs]))
        # x = torch.linspace(-3 * sigma, 3 * sigma, 10000, device=device)
        # y = integrand(x, sigma)
        # integral = torch.trapz(y, x)
        # # result = torch.exp(integral)
        # result = torch.log(integral)
        # reg_loss += result
        # del x, y, integral, result
        # torch.cuda.empty_cache()

    elif type == "R2":
        # C = args.clipping_bound
        # norm_diff = torch.sum(torch.stack([torch.norm(diff) for diff in param_diffs]))
        # reg_loss = (args.lambda_2 / 2) * torch.norm(norm_diff - C)
        reg_loss = 0

    else:
        raise ValueError("Invalid regularization type")

    return ce_loss + reg_loss

def local_update_fedavg(model, dataloader, global_model, client):
    model = model.to(device)
    global_model = global_model.to(device)

    w_glob = [param.clone().detach() for param in global_model.parameters()]

    for epoch in range(args.local_epoch):
        # w_last_round = [param.clone().detach() for param in model.parameters()]
        # 随机选取一个 batch
        batch = random.choice(list(dataloader))
        datas, labels = batch
        # 将batch中的每个数据单独处理
        batch_gradient = None
        for i in range(datas.size(0)):  # data.size(0)是batch的大小
            sample_data = datas[i].unsqueeze(0)  # 取出第i个样本，并保持维度一致
            sample_data = sample_data.to(device)
            sample_label = labels[i].unsqueeze(0)  # 取出对应的标签，并保持维度一致
            sample_label = sample_label.to(device)
            # optimizer1.zero_grad()  # 清空梯度
            output = model(sample_data)  # 前向传播
            loss = F.cross_entropy(output, sample_label)
            gradient = torch.autograd.grad(loss, model.parameters(), retain_graph=True, create_graph=True,
                                           only_inputs=True)
            norm = 0
            for grad in gradient:
                current_norm = torch.norm(grad, p=2)
                norm += torch.pow(current_norm, 2)
            clip_rate = max(1, (math.sqrt(norm) / clipping_bound))
            if batch_gradient is None:
                batch_gradient = [(grad / clip_rate) for grad in gradient]
            else:
                new_gradient = [(grad / clip_rate) for grad in gradient]
                for idx, (grad1, grad2) in enumerate(zip(batch_gradient, new_gradient)):
                    batch_gradient[idx] = grad1 + grad2
        batch_gradient = [(grad / datas.size(0)) for grad in batch_gradient]
        noisy_gradients = []
        for grad in batch_gradient:
            sigma = client.ba.noise_multiplier
            noise = torch.normal(mean=0.0, std=clipping_bound * sigma, size=grad.shape)
            noise = noise / datas.size(0)
            noise = noise.to(device)
            new_grad = grad + noise
            noisy_gradients.append(new_grad)
        # Update model weights with gradients and learning rate
        for param, grad_part in zip(model.parameters(), noisy_gradients):
            param.data = param.data - args.lr * grad_part
    client.ba.update(client.loc_steps)

    with torch.no_grad():
        update = [(new_param - old_param).clone() for new_param, old_param in zip(model.parameters(), w_glob)]
    model = model.to('cpu')
    global_model = global_model.to('cpu')
    return update


## 先根据对数概率计算fisher信息矩阵，进而划分ui和vi；
## 每个local_epoch取一个batch的训练数据，进行两步操作：
##      1.ui以 交叉熵损失+(ui-ui_last_round)的范数 计算梯度，默认裁剪范数，并添加高斯噪声，标准差std：default_clip_norm * init_nm，更新模型参数
##      2.vi以 交叉熵损失 计算梯度，默认裁剪范数，不添加噪声，更新模型参数
def local_update_first(model, dataloader, global_model, client):
    fisher_threshold = args.fisher_threshold
    model = model.to(device)
    global_model = global_model.to(device)

    w_glob = [param.clone().detach() for param in global_model.parameters()]

    fisher_diag = compute_fisher_diag(model, dataloader)

    u_loc, v_loc = [], []
    for param, fisher_value in zip(model.parameters(), fisher_diag):
        u_param = (param * (fisher_value > fisher_threshold)).clone().detach()
        v_param = (param * (fisher_value <= fisher_threshold)).clone().detach()
        u_loc.append(u_param)
        v_loc.append(v_param)

    # for u_param, fisher_value in zip(u_loc, fisher_diag):
    #     print('该层初始fisher和为：{}'.format(torch.sum(fisher_value)))
    #     print('该层平均参数fisher为：{}'.format(torch.sum(fisher_value) / fisher_value.numel()))
    #     print('该层ui的fisher和为：{}'.format(torch.sum(fisher_value * (u_param != 0))))
    #     print('该层ui的平均fisher和为：{}'.format(torch.sum(fisher_value * (u_param != 0)) / torch.nonzero(fisher_value * (u_param != 0)).size(0)))

    means = []
    for u_param, fisher_value in zip(u_loc, fisher_diag):
        meanl = torch.sum(fisher_value * (u_param != 0)) / torch.nonzero(fisher_value * (u_param != 0)).size(0)
        means.append(meanl)
    min_mean = min(means)

    for epoch in range(args.local_epoch):
        # w_last_round = [param.clone().detach() for param in model.parameters()]
        # 随机选取一个 batch
        batch = random.choice(list(dataloader))
        datas, labels = batch
        # 将batch中的每个数据单独处理
        batch_gradient = None
        for i in range(datas.size(0)):  # data.size(0)是batch的大小
            sample_data = datas[i].unsqueeze(0)  # 取出第i个样本，并保持维度一致
            sample_data = sample_data.to(device)
            sample_label = labels[i].unsqueeze(0)  # 取出对应的标签，并保持维度一致
            sample_label = sample_label.to(device)
            # optimizer1.zero_grad()  # 清空梯度
            output = model(sample_data)  # 前向传播
            param_diffs = [u_new - u_old for u_new, u_old in zip(model.parameters(), w_glob)]
            for idx, (param, u_param) in enumerate(zip(param_diffs, u_loc)):
                param_diffs[idx] = param * (u_param != 0)
            loss = customloss(output, sample_label, "R1", param_diffs, clipping_bound * client.ba.noise_multiplier)
            gradient = torch.autograd.grad(loss, model.parameters(), retain_graph=True, create_graph=True,
                                           only_inputs=True)
            norm = 0
            for grad in gradient:
                current_norm = torch.norm(grad, p=2)
                norm += torch.pow(current_norm, 2)
            clip_rate = max(1, (math.sqrt(norm) / clipping_bound))
            if batch_gradient is None:
                batch_gradient = [(grad / clip_rate) for grad in gradient]
            else:
                new_gradient = [(grad / clip_rate) for grad in gradient]
                for idx, (grad1, grad2) in enumerate(zip(batch_gradient, new_gradient)):
                    batch_gradient[idx] = grad1 + grad2
        batch_gradient = [(grad / datas.size(0)) for grad in batch_gradient]
        for idx, (grad, u_param) in enumerate(zip(batch_gradient, u_loc)):
            batch_gradient[idx] = grad * (u_param != 0)
        noisy_gradients = []
        ## myalgo 1,2
        # for grad, fisher_value in zip(batch_gradient, fisher_diag):
        #     noise = torch.normal(mean=0.0, std=clipping_bound * client.ba.noise_multiplier, size=grad.shape)
        #     noise = noise / datas.size(0)
        #     noise = noise.to(device)
        #     new_grad = grad + noise * (fisher_value > fisher_threshold)
        #     noisy_gradients.append(new_grad)
        ##  myalgo 论文
        for grad, fisher_value, meanl in zip(batch_gradient, fisher_diag, means):
            sigma = client.ba.noise_multiplier * (1 + (meanl-min_mean)/(min_mean * 10))
            noise = torch.normal(mean=0.0, std=clipping_bound * sigma, size=grad.shape)
            noise = noise / datas.size(0)
            noise = noise.to(device)
            new_grad = grad + noise * (fisher_value > fisher_threshold)
            noisy_gradients.append(new_grad)
        # Update model weights with gradients and learning rate
        for param, grad_part in zip(model.parameters(), noisy_gradients):
            param.data = param.data - args.lr * grad_part

    # optimizer2 = optim.SGD(model.parameters(), lr=args.lr)
    for epoch in range(args.local_epoch):
        # 随机选取一个 batch
        batch = random.choice(list(dataloader))
        datas, labels = batch
        # 将batch中的每个数据单独处理
        batch_gradient = None
        for i in range(datas.size(0)):  # data.size(0)是batch的大小
            sample_data = datas[i].unsqueeze(0)  # 取出第i个样本，并保持维度一致
            sample_data = sample_data.to(device)
            sample_label = labels[i].unsqueeze(0)  # 取出对应的标签，并保持维度一致
            sample_label = sample_label.to(device)
            # optimizer2.zero_grad()  # 清空梯度
            output = model(sample_data)  # 前向传播
            loss = customloss(output, sample_label, "R2")
            gradient = torch.autograd.grad(loss, model.parameters(), retain_graph=True, create_graph=True,
                                           only_inputs=True)
            norm = 0
            for grad in gradient:
                current_norm = torch.norm(grad, p=2)
                norm += torch.pow(current_norm, 2)
            clip_rate = max(1, (math.sqrt(norm) / clipping_bound))
            if batch_gradient is None:
                batch_gradient = [(grad / clip_rate) for grad in gradient]
            else:
                new_gradient = [(grad / clip_rate) for grad in gradient]
                for idx, (grad1, grad2) in enumerate(zip(batch_gradient, new_gradient)):
                    batch_gradient[idx] = grad1 + grad2
        batch_gradient = [(grad / datas.size(0)) for grad in batch_gradient]
        for idx, (grad, v_param) in enumerate(zip(batch_gradient, v_loc)):
            batch_gradient[idx] = grad * (v_param != 0)
        # Update model weights with gradients and learning rate
        for param, grad_part in zip(model.parameters(), batch_gradient):
            param.data = param.data - args.lr * grad_part

    client.ba.update(client.loc_steps)

    with torch.no_grad():
        update = [(new_param - old_param).clone() for new_param, old_param in zip(model.parameters(), w_glob)]
    model = model.to('cpu')
    global_model = global_model.to('cpu')
    return update


## 先根据对数概率计算fisher信息矩阵，进而划分ui和vi；
## 每个local_epoch取一个batch的训练数据，进行两步操作：
##      1.ui以 交叉熵损失+(ui-global_ui)的范数 计算梯度，
##      2.vi以 交叉熵损失 计算梯度，
##      将两组梯度按照层组合，每层自适应裁剪范数，并分层对ui的梯度添加高斯噪声，标准差std：clip_norm(k) * nm(k)
## 更新模型参数
def local_update_decay(model, dataloader, global_model, latest_global_model, client):
    fisher_threshold = args.fisher_threshold
    model = model.to(device)
    global_model = global_model.to(device)
    latest_global_model = latest_global_model.to(device)

    k = eps2level(client)
    w_glob = [param.clone().detach() for param in global_model.parameters()]
    w_latest = [param.clone().detach() for param in latest_global_model.parameters()]
    lowests = []
    highests = []
    # norms = []
    for global_para in w_latest:
        c = global_para.mean()
        min_value = global_para.min()
        max_value = global_para.max()
        r = max(abs(c - min_value), abs(max_value - c))
        lowest = - (c + k * r - global_para) / args.lr
        highest = - (c - k * r - global_para) / args.lr
        # norm_low = torch.norm(lowest, p=2)
        # norm_high = torch.norm(highest, p=2)
        # if norm_low > norm_high:
        #     norms.append(norm_low)
        # else:
        #     norms.append(norm_high)
        lowests.append(lowest)
        highests.append(highest)

    fisher_diag = compute_fisher_diag(model, dataloader)

    u_loc, v_loc = [], []
    for param, fisher_value in zip(model.parameters(), fisher_diag):
        u_param = (param * (fisher_value > fisher_threshold)).clone().detach()
        v_param = (param * (fisher_value <= fisher_threshold)).clone().detach()
        u_loc.append(u_param)
        v_loc.append(v_param)

    means = []
    for u_param, fisher_value in zip(u_loc, fisher_diag):
        meanl = torch.sum(fisher_value * (u_param != 0)) / torch.nonzero(fisher_value * (u_param != 0)).size(0)
        means.append(meanl)
    min_mean = min(means)

    loss_sigma = None
    for epoch in range(args.local_epoch):
        # w_last_round = [param.clone().detach() for param in model.parameters()]
        # 随机选取一个 batch
        batch = random.choice(list(dataloader))
        datas, labels = batch
        # 将batch中的每个数据单独处理
        batch_gradient = None
        norms = []
        for i in range(datas.size(0)):  # data.size(0)是batch的大小
            sample_data = datas[i].unsqueeze(0)  # 取出第i个样本，并保持维度一致
            sample_data = sample_data.to(device)
            sample_label = labels[i].unsqueeze(0)  # 取出对应的标签，并保持维度一致
            sample_label = sample_label.to(device)
            # optimizer1.zero_grad()  # 清空梯度
            output = model(sample_data)  # 前向传播

            param_diffs = [u_new - u_old for u_new, u_old in zip(model.parameters(), w_glob)]
            for idx, (param, u_param) in enumerate(zip(param_diffs, u_loc)):
                param_diffs[idx] = param * (u_param != 0)
            if loss_sigma is None:
                sigma = clipping_bound * client.ba.noise_multiplier
            else:
                sigma = loss_sigma
            loss1 = customloss(output, sample_label, "R1", param_diffs, sigma)
            gradient1 = torch.autograd.grad(loss1, model.parameters(), retain_graph=True, create_graph=True,
                                            only_inputs=True)
            gradient1 = list(gradient1)
            for idx, (grad, u_param) in enumerate(zip(gradient1, u_loc)):
                gradient1[idx] = grad * (u_param != 0)

            loss2 = customloss(output, sample_label, "R2")
            gradient2 = torch.autograd.grad(loss2, model.parameters(), retain_graph=True, create_graph=True,
                                            only_inputs=True)
            gradient2 = list(gradient2)
            for idx, (grad, v_param) in enumerate(zip(gradient2, v_loc)):
                gradient2[idx] = grad * (v_param != 0)
            gradient = [grad1 + grad2 for grad1, grad2 in zip(gradient1, gradient2)]

            for idx, (grad, lowest, highest) in enumerate(zip(gradient, lowests, highests)):
                # current_grad = torch.max(grad, lowest)
                # current_grad = torch.min(current_grad, highest)
                current_grad = torch.min(grad, highest)
                if torch.norm(current_grad, p=2) > 4.0:
                    clip_rate = max(1.0, torch.norm(current_grad, p=2).item() / 4.0)
                    current_grad = current_grad / clip_rate
                gradient[idx] = current_grad
                if (i == 0) and (torch.norm(current_grad, p=2) < 0.5):
                    norms.append(torch.tensor(0.5, device=device))
                elif (i == 0) and (torch.norm(current_grad, p=2) >= 0.5):
                    norms.append(torch.norm(current_grad, p=2))
                else:
                    if torch.norm(current_grad, p=2) > norms[idx]:
                        norms[idx] = torch.norm(current_grad, p=2)
            if batch_gradient is None:
                batch_gradient = gradient
            else:
                for idx, (grad1, grad2) in enumerate(zip(batch_gradient, gradient)):
                    batch_gradient[idx] = grad1 + grad2
        batch_gradient = [(grad / datas.size(0)) for grad in batch_gradient]
        noisy_gradients = []
        for grad, fisher_value, meanl, norm in zip(batch_gradient, fisher_diag, means, norms):
            sigma = client.ba.noise_multiplier * (1 + (meanl-min_mean)/(min_mean * 10))
            std = (norm * sigma).item()
            if np.isnan(std):
                std = 0.5
            noise = torch.normal(mean=0.0, std=std, size=grad.shape)
            noise = noise / datas.size(0)
            noise = noise.to(device)
            new_grad = grad + noise * (fisher_value > fisher_threshold)
            noisy_gradients.append(new_grad)
        # Update model weights with gradients and learning rate
        for param, grad_part in zip(model.parameters(), noisy_gradients):
            param.data = param.data - args.lr * grad_part
        loss_sigma = min(norms) * client.ba.noise_multiplier
    client.ba.update(client.loc_steps)

    with torch.no_grad():
        update = [(new_param - old_param).clone() for new_param, old_param in zip(model.parameters(), w_glob)]
    model.to('cpu')
    global_model.to('cpu')
    latest_global_model.to('cpu')
    return update


def test(client_model, client_testloader):
    client_model.eval()
    client_model = client_model.to(device)

    num_data = 0

    correct = 0
    with torch.no_grad():
        for data, labels in client_testloader:
            data, labels = data.to(device), labels.to(device)
            outputs = client_model(data)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            num_data += labels.size(0)

    accuracy = 100.0 * correct / num_data

    client_model.to('cpu')

    return accuracy


def aggregate(client_updates, sampled_client_data_sizes, sampled_client_eps, fedavg=False, weiavg=False, deavg=True):
    if fedavg:
        sampled_client_weights = [sampled_client_data_size / sum(sampled_client_data_sizes)
                                  for sampled_client_data_size in sampled_client_data_sizes]
    elif weiavg:
        sampled_client_weights = [sampled_client_e / sum(sampled_client_eps)
            for sampled_client_e in sampled_client_eps]
    elif deavg:
        weights1 = [sampled_client_data_size / sum(sampled_client_data_sizes)
            for sampled_client_data_size in sampled_client_data_sizes]
        eps_sum = 0.0
        for eps in sampled_client_eps:
            eps_sum += np.e ** eps
        weights2 = [np.e ** eps /eps_sum for eps in sampled_client_eps]
        weights1 = np.array(weights1)
        weights2 = np.array(weights2)
        sampled_client_weights = (1 - alpha) * weights1 + alpha * weights2
    else:
        raise ValueError('No aggregate algo defined!')

    aggregated_update = [
        torch.sum(
            torch.stack(
                [
                    noisy_update[param_index] * sampled_client_weights[idx]
                    for idx, noisy_update in enumerate(client_updates)
                ]
            ),
            dim=0,
        )
        for param_index in range(len(client_updates[0]))
    ]

    return aggregated_update


def main():
    for time_ in range(1):
        print(f"num_client: {num_clients}, time: {time_+1}")
        mean_acc_s = []
        acc_matrix = []
        global_acc = []

        ## get data and model
        if dataset == 'mnist':
            train_dataset, test_dataset = get_mnist_datasets()
            clients_train_sets = get_clients_datasets(train_dataset, num_clients)
            client_data_sizes = [len(client_dataset) for client_dataset in clients_train_sets]
            clients_train_loaders = [DataLoader(client_dataset, batch_size=batch_size) for client_dataset in
                                         clients_train_sets]
            clients_test_loaders = [DataLoader(test_dataset) for i in range(num_clients)]

            clients_models = [mnistNet() for _ in range(num_clients)]
            global_model = mnistNet()
            # num_classes = 10  # mnist数据分类为十分类： 0 ～ 9
            # channel = 1  # mnist数据集是灰度图像所以是单通道
            # hidden = 588  # hidden是神经网络最后一层全连接层的维度
            # clients_models = [LeNet(channel=channel, hidden=hidden, num_classes=num_classes) for _ in range(num_clients)]
            # global_model = LeNet(channel=channel, hidden=hidden, num_classes=num_classes)
        elif dataset == 'fmnist':
            # train_dataset, test_dataset = get_fmnist_datasets()
            # clients_train_sets = get_clients_datasets(train_dataset, num_clients)
            # client_data_sizes = [len(client_dataset) for client_dataset in clients_train_sets]
            # clients_train_loaders = [DataLoader(client_dataset, batch_size=batch_size) for client_dataset in
            #                              clients_train_sets]
            # clients_test_loaders = [DataLoader(test_dataset) for i in range(num_clients)]
            ## noniid-fmnist
            clients_train_loaders, clients_test_loaders, client_data_sizes = get_noniid_fmnist(args.dir_alpha, num_clients)

            clients_models = [fmnistNet() for _ in range(num_clients)]
            global_model = fmnistNet()
            # num_classes = 10  # mnist数据分类为十分类： 0 ～ 9
            # channel = 1  # mnist数据集是灰度图像所以是单通道
            # hidden = 588  # hidden是神经网络最后一层全连接层的维度
            # clients_models = [LeNet(channel=channel, hidden=hidden, num_classes=num_classes) for _ in range(num_clients)]
            # global_model = LeNet(channel=channel, hidden=hidden, num_classes=num_classes)
        elif dataset == 'cifar10':
            clients_train_loaders, clients_test_loaders, client_data_sizes = get_CIFAR10(args.dir_alpha, num_clients)

            clients_models = [cifarNet() for _ in range(num_clients)]
            global_model = cifarNet()
            # clients_models = [LeNet() for _ in range(num_clients)]
            # global_model = LeNet()
        else:
            raise ValueError('undifined dataset')

        for client_model in clients_models:
            client_model.load_state_dict(global_model.state_dict())
        ## get epsilon
        priv_preferences = set_epsilons(target_epsilon, num_clients)
        priv_preferences = np.array(priv_preferences)
        clients = []
        for cid in range(num_clients):
            client = Client(train_data=clients_train_loaders[cid],
                                test_data=clients_test_loaders[cid],
                                batch_size=batch_size,
                                model=clients_models[cid],
                                loc_steps=local_epoch,
                                data_size=client_data_sizes[cid])
            client_eps = priv_preferences[cid]
            if nm_decay:
                nm = compute_noise_multiplier_decay(target_epsilon=client_eps, target_delta=target_delta,
                                                        global_epoch=global_epoch*user_sample_rate, local_steps=local_epoch,
                                                        L=batch_size, N=client_data_sizes[cid], decay_factor=decay_factor)
                # nm = compute_noise_multiplier(N=client_data_sizes[cid], L=batch_size, epsilon=client_eps,
                #                               delta=target_delta,
                #                               T=global_epoch * local_epoch * user_sample_rate)

            else:
                nm = compute_noise_multiplier(N=client_data_sizes[cid], L=batch_size, epsilon=client_eps, delta=target_delta,
                                                  T=global_epoch*local_epoch*user_sample_rate)
            print(f"initial nm:{nm}")
            ba = MomentsAccountant(epsilon=client_eps, delta=target_delta, noise_multiplier=nm)
            client.set_ba(ba)

            clients.append(client)

        latest_global_model = None

        # ##DLG
        # start_idx = [0]
        # for i in range(1, num_clients):
        #     start_idx.append(start_idx[-1] + client_data_sizes[i - 1])
        # print(f"start_idx: {start_idx}")

        for epoch in trange(global_epoch):
            # precheck and pick up the candidates who can take the next commiunication round.
            candidates = [cid for cid in range(num_clients) if clients[cid].precheck()]
            if len(candidates) < int(user_sample_rate * num_clients):
                print('There are no enough clients can be trained!')
                break
            else:
                sampled_client_indices = random.sample(candidates, max(1, int(user_sample_rate * num_clients)))
                sampled_clients_models = [clients_models[i] for i in sampled_client_indices]
                sampled_clients_train_loaders = [clients_train_loaders[i] for i in sampled_client_indices]
                sampled_clients_test_loaders = [clients_test_loaders[i] for i in sampled_client_indices]
                sampled_clients = [clients[i] for i in sampled_client_indices]
                # ##DLG
                # sampled_clients_idx = [start_idx[i] for i in sampled_client_indices]

                # download global model
                for client_model in sampled_clients_models:
                    client_model.load_state_dict(global_model.state_dict())
                clients_model_updates = []
                clients_accuracies = []
                st_time = time.time()
                for idx, (client, client_model, client_trainloader, client_testloader) in enumerate(
                            zip(sampled_clients, sampled_clients_models, sampled_clients_train_loaders, sampled_clients_test_loaders)):
                    if latest_global_model is None:
                        client_update = local_update_first(model=client_model, dataloader=client_trainloader,
                                                               global_model=global_model,
                                                               client=client)
                    else:
                        client_update = local_update_decay(model=client_model, dataloader=client_trainloader,
                                                               global_model=global_model,
                                                               latest_global_model=latest_global_model,
                                                               client=client)
                    # client_update = local_update_fedavg(model=client_model, dataloader=client_trainloader,
                    #                                     global_model=global_model,
                    #                                     client=client)
                    clients_model_updates.append(client_update)
                    accuracy = test(client_model, client_testloader)
                    clients_accuracies.append(accuracy)
                # if latest_global_model is None:
                #     client_update = local_update_first(model=clients_models[0], dataloader=clients_train_loaders[0],
                #                                        global_model=global_model,
                #                                        client=clients[0])
                # else:
                #     client_update = local_update_decay(model=clients_models[0], dataloader=clients_train_loaders[0],
                #                                        global_model=global_model,
                #                                        latest_global_model=latest_global_model,
                #                                        client=clients[0])
                # accuracy = test(clients_models[0], clients_test_loaders[0])
                # clients_accuracies.append(accuracy)
                print(clients_accuracies)
                # ##DLG
                # if epoch == 19:
                #     dlg_attack(sampled_clients_models[0], dataset, epoch, idx=1)

                mean_acc_s.append(sum(clients_accuracies) / len(clients_accuracies))
                acc_matrix.append(clients_accuracies)
                sampled_client_data_sizes = [client_data_sizes[i] for i in sampled_client_indices]
                sampled_client_eps = [priv_preferences[i] for i in sampled_client_indices]

                aggregated_update = aggregate(client_updates=clients_model_updates,
                                                  sampled_client_data_sizes=sampled_client_data_sizes,
                                                  sampled_client_eps=sampled_client_eps,
                                                  fedavg=fedavg, weiavg=weiavg, deavg=deavg)
                with torch.no_grad():
                    global_model = global_model.to(device)
                    for global_param, update in zip(global_model.parameters(), aggregated_update):
                        global_param.add_(update)
                en_time = time.time()
                print(f"cost time:{en_time-st_time}")
                global_accuracy = test(global_model, clients_test_loaders[0])
                if (epoch >= 2) and (global_accuracy >= global_acc[-1]) and (global_acc[-1] >= global_acc[-2]) and all(global_accuracy > x for x in global_acc):
                    latest_global_model = copy.deepcopy(global_model)
                    for client in sampled_clients:
                        client.ba.noise_multiplier *= decay_factor
                print('epoch:{}, global accuracy:{}'.format(epoch+1, global_accuracy))
                global_acc.append(global_accuracy)

        acc = pd.DataFrame(global_acc)
        file_name = (
                f'./{dataset}/'
                f'scen3_AdapL_'
                f'{dataset}_'
                f'numclients_{num_clients}_without2.csv'
            )
        acc.to_csv(file_name, index=False, header=None)
        char_set = '1234567890abcdefghijklmnopqrstuvwxyz'
        ID = ''
        for ch in random.sample(char_set, 5):
            ID = f'{ID}{ch}'
        print(
                f'===============================================================\n'
                f'task_ID : '
                f'{ID}\n'
                f'main_yxy\n'
                f'mean accuracy : \n'
                f'{mean_acc_s}\n'
                f'acc matrix : \n'
                f'{torch.tensor(acc_matrix)}\n'
                f'global accuracy : \n'
                f'{global_acc}\n'
                f'===============================================================\n'
            )

if __name__ == '__main__':
        main()