import os
import sys
import random
import time
from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch
import copy
import math
from torch.utils.data import DataLoader
from data import get_mnist_datasets, get_clients_datasets, get_fmnist_datasets, get_cifar10_datasets, get_cifar100_datasets, get_CIFAR10, get_CIFAR100, get_noniid_fmnist
from model import *
from client import Client
from dpsgd_utils import *
from utils import *
from tqdm.auto import trange, tqdm
from options import parse_args
import torch.optim as optim
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

args = parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

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

if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    device = torch.device("cuda")
    print(f"Using GPU: cuda:{args.device}")
else:
    device = torch.device("cpu")
    print("GPU not available, using CPU")

if args.store:
    saved_stdout = sys.stdout
    # 构建目录路径
    dir_path = BASE_DIR / 'txt' / target_epsilon

    # 如果目录不存在，创建目录
    os.makedirs(dir_path, exist_ok=True)

    # 构建文件路径和文件名
    file_name = (
        f'{dir_path}/'
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


def get_important_fisher_mean(fisher_value, important_mask):
    selected = fisher_value[important_mask]
    if selected.numel() == 0:
        return torch.tensor(0.0, device=fisher_value.device)
    mean_value = selected.mean()
    if not torch.isfinite(mean_value):
        return torch.tensor(0.0, device=fisher_value.device)
    return mean_value


def get_layer_noise_multiplier(base_sigma, mean_value, min_mean):
    if not torch.isfinite(mean_value) or not torch.isfinite(min_mean) or min_mean.item() <= 0:
        return base_sigma
    scale = 1 + ((mean_value - min_mean) / (min_mean * args.gamma)).item()
    if not np.isfinite(scale) or scale < 0:
        return base_sigma
    return base_sigma * scale


def get_clip_rate(norm, bound):
    norm_value = math.sqrt(norm.detach().item())
    return max(1.0, norm_value / bound)


def move_batch_to_device(datas, labels):
    use_non_blocking = torch.cuda.is_available()
    return datas.to(device, non_blocking=use_non_blocking), labels.to(device, non_blocking=use_non_blocking)


def get_masked_param_diffs(model, reference_params, important_masks):
    return [
        (param - reference_param) * important_mask
        for param, reference_param, important_mask in zip(model.parameters(), reference_params, important_masks)
    ]


def get_regularization_gradient(model, param_diffs):
    reg_loss = torch.sum(torch.stack([torch.norm(diff) for diff in param_diffs]))
    return list(torch.autograd.grad(reg_loss, model.parameters(), only_inputs=True))


def get_fisher_batch_limit(dataloader):
    if args.fisher_max_batches > 0:
        return min(args.fisher_max_batches, len(dataloader))
    return len(dataloader)


def iter_local_batches(dataloader, local_steps):
    train_iter = iter(dataloader)
    for _ in range(local_steps):
        try:
            yield next(train_iter)
        except StopIteration:
            train_iter = iter(dataloader)
            yield next(train_iter)


def init_momentum_buffers(model):
    if args.momentum <= 0:
        return None
    return [torch.zeros_like(param, device=param.device) for param in model.parameters()]


def apply_local_gradients(model, gradients, momentum_buffers=None):
    with torch.no_grad():
        for idx, (param, grad) in enumerate(zip(model.parameters(), gradients)):
            update = grad.detach()
            if args.weight_decay != 0:
                update = update + args.weight_decay * param.data
            if momentum_buffers is not None:
                momentum_buffers[idx].mul_(args.momentum).add_(update)
                update = momentum_buffers[idx]
            param.data = param.data - args.lr * update


def get_selected_client_count():
    if not 0 < user_sample_rate <= 1:
        raise ValueError("--user_sample_rate/--client_fraction must be in (0, 1].")
    return max(1, int(math.ceil(num_clients * user_sample_rate)))


def select_clients(candidates, round_idx):
    candidate_set = set(candidates)
    generator = torch.Generator().manual_seed(args.seed + round_idx)
    permutation = torch.randperm(num_clients, generator=generator).tolist()
    selected = [cid for cid in permutation if cid in candidate_set][:get_selected_client_count()]
    return sorted(selected)


def make_train_loader(client_dataset, client_id):
    generator = torch.Generator().manual_seed(args.seed + client_id)
    worker_kwargs = {}
    if args.num_workers > 0:
        worker_kwargs["prefetch_factor"] = args.prefetch_factor
        worker_kwargs["persistent_workers"] = args.persistent_workers
    return DataLoader(
        client_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
        **worker_kwargs,
    )


def make_shared_test_loaders(test_dataset):
    worker_kwargs = {}
    if args.num_workers > 0:
        worker_kwargs["prefetch_factor"] = args.prefetch_factor
        worker_kwargs["persistent_workers"] = args.persistent_workers
    shared_test_loader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        **worker_kwargs,
    )
    return [shared_test_loader for _ in range(num_clients)]


def local_update_fedavg(model, dataloader, global_model, client):
    model = model.to(device)
    global_model = global_model.to(device)

    w_glob = [param.clone().detach() for param in global_model.parameters()]
    momentum_buffers = init_momentum_buffers(model)

    model.train()
    for datas, labels in iter_local_batches(dataloader, args.local_epoch):
        datas, labels = move_batch_to_device(datas, labels)
        # w_last_round = [param.clone().detach() for param in model.parameters()]
        # 将batch中的每个数据单独处理
        batch_gradient = None
        for i in range(datas.size(0)):  # data.size(0)是batch的大小
            sample_data = datas[i].unsqueeze(0)  # 取出第i个样本，并保持维度一致
            sample_label = labels[i].unsqueeze(0)  # 取出对应的标签，并保持维度一致
            # optimizer1.zero_grad()  # 清空梯度
            output = model(sample_data)  # 前向传播
            loss = F.cross_entropy(output, sample_label)
            gradient = torch.autograd.grad(loss, model.parameters(), only_inputs=True)
            norm = 0
            for grad in gradient:
                current_norm = torch.norm(grad, p=2)
                norm += torch.pow(current_norm, 2)
            clip_rate = get_clip_rate(norm, clipping_bound)
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
            noise = torch.randn_like(grad) * (clipping_bound * sigma)
            noise = noise / datas.size(0)
            new_grad = grad + noise
            noisy_gradients.append(new_grad)
        # Update model weights with gradients and learning rate
        apply_local_gradients(model, noisy_gradients, momentum_buffers)
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

    if args.verbose_logs:
        print(
            f"Computing Fisher: data_size={client.data_size}, "
            f"batches={get_fisher_batch_limit(dataloader)}/{len(dataloader)}, "
            f"estimator={args.fisher_estimator}",
            flush=True,
        )
    fisher_diag = compute_fisher_diag(model, dataloader, args.fisher_max_batches, args.fisher_estimator)
    if args.verbose_logs:
        print(f"Finished Fisher: data_size={client.data_size}", flush=True)

    important_masks = [fisher_value > fisher_threshold for fisher_value in fisher_diag]

    # for u_param, fisher_value in zip(u_loc, fisher_diag):
    #     print('该层初始fisher和为：{}'.format(torch.sum(fisher_value)))
    #     print('该层平均参数fisher为：{}'.format(torch.sum(fisher_value) / fisher_value.numel()))
    #     print('该层ui的fisher和为：{}'.format(torch.sum(fisher_value * (u_param != 0))))
    #     print('该层ui的平均fisher和为：{}'.format(torch.sum(fisher_value * (u_param != 0)) / torch.nonzero(fisher_value * (u_param != 0)).size(0)))

    means = []
    for important_mask, fisher_value in zip(important_masks, fisher_diag):
        meanl = get_important_fisher_mean(fisher_value, important_mask)
        means.append(meanl)
    min_mean = torch.min(torch.stack(means))

    model.train()
    momentum_buffers = init_momentum_buffers(model)
    for datas, labels in iter_local_batches(dataloader, args.local_epoch):
        datas, labels = move_batch_to_device(datas, labels)
        param_diffs = get_masked_param_diffs(model, w_glob, important_masks)
        regularization_gradient = get_regularization_gradient(model, param_diffs)
        batch_gradient = None
        for i in range(datas.size(0)):
            sample_data = datas[i].unsqueeze(0)
            sample_label = labels[i].unsqueeze(0)
            output = model(sample_data)

            loss = customloss(output, sample_label, "R2")
            gradient = torch.autograd.grad(loss, model.parameters(), only_inputs=True)
            gradient = [
                grad + reg_grad * important_mask
                for grad, reg_grad, important_mask in zip(gradient, regularization_gradient, important_masks)
            ]
            norm = 0
            for grad in gradient:
                current_norm = torch.norm(grad, p=2)
                norm += torch.pow(current_norm, 2)
            clip_rate = get_clip_rate(norm, clipping_bound)
            clipped_gradient = [(grad / clip_rate) for grad in gradient]

            if batch_gradient is None:
                batch_gradient = clipped_gradient
            else:
                for idx, (grad1, grad2) in enumerate(zip(batch_gradient, clipped_gradient)):
                    batch_gradient[idx] = grad1 + grad2

        batch_gradient = [(grad / datas.size(0)) for grad in batch_gradient]
        noisy_gradients = []
        for grad, important_mask, meanl in zip(batch_gradient, important_masks, means):
            sigma = get_layer_noise_multiplier(client.ba.noise_multiplier, meanl, min_mean)
            noise = torch.randn_like(grad) * (clipping_bound * sigma)
            noise = noise / datas.size(0)
            new_grad = grad + noise * important_mask
            noisy_gradients.append(new_grad)
        apply_local_gradients(model, noisy_gradients, momentum_buffers)

    client.ba.update(client.loc_steps)

    model = model.to('cpu')
    global_model = global_model.to('cpu')
    return None


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

    if args.verbose_logs:
        print(
            f"Computing Fisher: data_size={client.data_size}, "
            f"batches={get_fisher_batch_limit(dataloader)}/{len(dataloader)}, "
            f"estimator={args.fisher_estimator}",
            flush=True,
        )
    fisher_diag = compute_fisher_diag(model, dataloader, args.fisher_max_batches, args.fisher_estimator)
    if args.verbose_logs:
        print(f"Finished Fisher: data_size={client.data_size}", flush=True)

    important_masks = [fisher_value > fisher_threshold for fisher_value in fisher_diag]

    means = []
    for important_mask, fisher_value in zip(important_masks, fisher_diag):
        meanl = get_important_fisher_mean(fisher_value, important_mask)
        means.append(meanl)
    min_mean = torch.min(torch.stack(means))

    model.train()
    momentum_buffers = init_momentum_buffers(model)
    for datas, labels in iter_local_batches(dataloader, args.local_epoch):
        datas, labels = move_batch_to_device(datas, labels)
        param_diffs = get_masked_param_diffs(model, w_glob, important_masks)
        regularization_gradient = get_regularization_gradient(model, param_diffs)
        # w_last_round = [param.clone().detach() for param in model.parameters()]
        # 将batch中的每个数据单独处理
        batch_gradient = None
        norms = []
        for i in range(datas.size(0)):  # data.size(0)是batch的大小
            sample_data = datas[i].unsqueeze(0)  # 取出第i个样本，并保持维度一致
            sample_label = labels[i].unsqueeze(0)  # 取出对应的标签，并保持维度一致
            # optimizer1.zero_grad()  # 清空梯度
            output = model(sample_data)  # 前向传播

            loss = customloss(output, sample_label, "R2")
            gradient = torch.autograd.grad(loss, model.parameters(), only_inputs=True)
            gradient = [
                grad + reg_grad * important_mask
                for grad, reg_grad, important_mask in zip(gradient, regularization_gradient, important_masks)
            ]

            for idx, (grad, lowest, highest) in enumerate(zip(gradient, lowests, highests)):
                current_grad = torch.max(grad, lowest)
                current_grad = torch.min(current_grad, highest)
                current_norm = torch.norm(current_grad, p=2)
                if current_norm > args.max_clip_norm:
                    clip_rate = max(1.0, current_norm.item() / args.max_clip_norm)
                    current_grad = current_grad / clip_rate
                    current_norm = torch.norm(current_grad, p=2)
                gradient[idx] = current_grad
                if (i == 0) and (current_norm < 0.5):
                    norms.append(torch.tensor(0.5, device=device))
                elif i == 0:
                    norms.append(current_norm)
                else:
                    if current_norm > norms[idx]:
                        norms[idx] = current_norm
            if batch_gradient is None:
                batch_gradient = gradient
            else:
                for idx, (grad1, grad2) in enumerate(zip(batch_gradient, gradient)):
                    batch_gradient[idx] = grad1 + grad2
        batch_gradient = [(grad / datas.size(0)) for grad in batch_gradient]
        noisy_gradients = []
        for grad, important_mask, meanl, norm in zip(batch_gradient, important_masks, means, norms):
            sigma = get_layer_noise_multiplier(client.ba.noise_multiplier, meanl, min_mean)
            std = (norm * sigma).item()
            if np.isnan(std) or std < 0:
                std = 0.5
            noise = torch.randn_like(grad) * std
            noise = noise / datas.size(0)
            new_grad = grad + noise * important_mask
            noisy_gradients.append(new_grad)
        # Update model weights with gradients and learning rate
        apply_local_gradients(model, noisy_gradients, momentum_buffers)
    client.ba.update(client.loc_steps)

    model.to('cpu')
    global_model.to('cpu')
    latest_global_model.to('cpu')
    return None


def evaluate(client_model, client_testloader):
    client_model.eval()
    client_model = client_model.to(device)

    num_data = 0

    correct = 0
    total_loss = 0.0
    with torch.no_grad():
        for data, labels in client_testloader:
            data, labels = move_batch_to_device(data, labels)
            outputs = client_model(data)
            total_loss += F.cross_entropy(outputs, labels, reduction='sum').item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            num_data += labels.size(0)

    accuracy = 100.0 * correct / num_data
    loss = total_loss / num_data

    client_model.to('cpu')

    return accuracy, loss


def test(client_model, client_testloader):
    accuracy, _ = evaluate(client_model, client_testloader)
    return accuracy


def get_method_name():
    if fedavg:
        return "FedAvg"
    if weiavg:
        return "WeiAvg"
    if deavg:
        return "AdapL"
    return "Unknown"


def format_arg_value(value):
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def get_model_name():
    model_names = {
        "mnist": "mnistnet",
        "fmnist": "fmnistnet",
        "cifar10": "cifarnet",
        "cifar100": "resnet18",
    }
    return model_names[dataset]


def get_partition_name():
    if args.iid:
        return "iid"
    return f"noniid_alpha{format_arg_value(args.dir_alpha)}"


def resolve_output_path(path_value):
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = BASE_DIR / path
    return path.resolve()


def build_result_csv_path():
    if args.output_csv:
        csv_path = resolve_output_path(args.output_csv)
        os.makedirs(csv_path.parent, exist_ok=True)
        return csv_path

    if args.output_dir:
        csv_dir = resolve_output_path(args.output_dir)
        file_name = (
            f"pf_{get_method_name().lower()}_"
            f"{dataset}_"
            f"{get_model_name()}_"
            f"{get_partition_name()}_"
            f"{args.privacy_scenario}_"
            f"k{num_clients}_"
            f"sr{format_arg_value(user_sample_rate)}_"
            f"steps{local_epoch}_"
            f"b{batch_size}_"
            f"lr{format_arg_value(args.lr)}_"
            f"r{global_epoch}.csv"
        )
    else:
        csv_dir = BASE_DIR / "results" / dataset
        file_name = f"scen3_AdapL_{dataset}_numclients_{num_clients}_without2.csv"

    os.makedirs(csv_dir, exist_ok=True)
    return csv_dir / file_name


def get_sampled_client_weights(sampled_client_data_sizes, sampled_client_eps, fedavg=False, weiavg=False, deavg=True):
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

    return sampled_client_weights


def aggregate(client_updates, sampled_client_data_sizes, sampled_client_eps, fedavg=False, weiavg=False, deavg=True):
    sampled_client_weights = get_sampled_client_weights(
        sampled_client_data_sizes,
        sampled_client_eps,
        fedavg=fedavg,
        weiavg=weiavg,
        deavg=deavg,
    )

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


def clone_state_dict(state_dict):
    return OrderedDict((name, tensor.detach().cpu().clone()) for name, tensor in state_dict.items())


def aggregate_client_states(client_states, sampled_client_data_sizes, sampled_client_eps, fedavg=False, weiavg=False, deavg=True):
    if not client_states:
        raise ValueError("Cannot aggregate an empty client state list.")

    sampled_client_weights = get_sampled_client_weights(
        sampled_client_data_sizes,
        sampled_client_eps,
        fedavg=fedavg,
        weiavg=weiavg,
        deavg=deavg,
    )

    aggregated_state = OrderedDict()
    for name in client_states[0].keys():
        first_value = client_states[0][name]
        if torch.is_floating_point(first_value):
            value = torch.zeros_like(first_value)
            for state, weight in zip(client_states, sampled_client_weights):
                value += state[name] * float(weight)
            aggregated_state[name] = value
        else:
            aggregated_state[name] = first_value.clone()

    return aggregated_state


def main():
    for time_ in range(1):
        print(f"num_client: {num_clients}, time: {time_+1}")
        mean_acc_s = []
        acc_matrix = []
        global_acc = []
        global_loss = []

        ## get data and model
        if dataset == 'mnist':
            train_dataset, test_dataset = get_mnist_datasets()
            clients_train_sets = get_clients_datasets(train_dataset, num_clients)
            client_data_sizes = [len(client_dataset) for client_dataset in clients_train_sets]
            clients_train_loaders = [
                make_train_loader(client_dataset, cid)
                for cid, client_dataset in enumerate(clients_train_sets)
            ]
            clients_test_loaders = make_shared_test_loaders(test_dataset)

            clients_models = [mnistNet() for _ in range(num_clients)]
            global_model = mnistNet()
            # num_classes = 10  # mnist数据分类为十分类： 0 ～ 9
            # channel = 1  # mnist数据集是灰度图像所以是单通道
            # hidden = 588  # hidden是神经网络最后一层全连接层的维度
            # clients_models = [LeNet(channel=channel, hidden=hidden, num_classes=num_classes) for _ in range(num_clients)]
            # global_model = LeNet(channel=channel, hidden=hidden, num_classes=num_classes)
        elif dataset == 'fmnist':
            if args.iid:
                train_dataset, test_dataset = get_fmnist_datasets()
                clients_train_sets = get_clients_datasets(train_dataset, num_clients)
                client_data_sizes = [len(client_dataset) for client_dataset in clients_train_sets]
                clients_train_loaders = [
                    make_train_loader(client_dataset, cid)
                    for cid, client_dataset in enumerate(clients_train_sets)
                ]
                clients_test_loaders = make_shared_test_loaders(test_dataset)
            else:
                clients_train_loaders, clients_test_loaders, client_data_sizes = get_noniid_fmnist(args.dir_alpha, num_clients)

            clients_models = [fmnistNet() for _ in range(num_clients)]
            global_model = fmnistNet()
            # num_classes = 10  # mnist数据分类为十分类： 0 ～ 9
            # channel = 1  # mnist数据集是灰度图像所以是单通道
            # hidden = 588  # hidden是神经网络最后一层全连接层的维度
            # clients_models = [LeNet(channel=channel, hidden=hidden, num_classes=num_classes) for _ in range(num_clients)]
            # global_model = LeNet(channel=channel, hidden=hidden, num_classes=num_classes)
        elif dataset == 'cifar10':
            if args.iid:
                train_dataset, test_dataset = get_cifar10_datasets()
                clients_train_sets = get_clients_datasets(train_dataset, num_clients)
                client_data_sizes = [len(client_dataset) for client_dataset in clients_train_sets]
                clients_train_loaders = [
                    make_train_loader(client_dataset, cid)
                    for cid, client_dataset in enumerate(clients_train_sets)
                ]
                clients_test_loaders = make_shared_test_loaders(test_dataset)
            else:
                clients_train_loaders, clients_test_loaders, client_data_sizes = get_CIFAR10(args.dir_alpha, num_clients)

            clients_models = [cifarNet() for _ in range(num_clients)]
            global_model = cifarNet()
            # clients_models = [LeNet() for _ in range(num_clients)]
            # global_model = LeNet()
        elif dataset == 'cifar100':
            if args.iid:
                train_dataset, test_dataset = get_cifar100_datasets()
                clients_train_sets = get_clients_datasets(train_dataset, num_clients)
                client_data_sizes = [len(client_dataset) for client_dataset in clients_train_sets]
                clients_train_loaders = [
                    make_train_loader(client_dataset, cid)
                    for cid, client_dataset in enumerate(clients_train_sets)
                ]
                clients_test_loaders = make_shared_test_loaders(test_dataset)
            else:
                clients_train_loaders, clients_test_loaders, client_data_sizes = get_CIFAR100(args.dir_alpha, num_clients)

            clients_models = [cifar100ResNet18() for _ in range(num_clients)]
            global_model = cifar100ResNet18()
        else:
            raise ValueError('undifined dataset')

        for client_model in clients_models:
            client_model.load_state_dict(global_model.state_dict())
        ## get epsilon
        priv_preferences = set_epsilons(target_epsilon, num_clients, args.privacy_scenario)
        priv_preferences = np.array(priv_preferences)
        clients = []
        noise_multipliers = []
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
            noise_multipliers.append(nm)
            if args.verbose_logs:
                print(f"initial nm:{nm}")
            ba = MomentsAccountant(epsilon=client_eps, delta=target_delta, noise_multiplier=nm)
            client.set_ba(ba)

            clients.append(client)
        if not args.verbose_logs:
            print(
                "initial nm summary: "
                f"min={np.min(noise_multipliers):.4f}, "
                f"max={np.max(noise_multipliers):.4f}, "
                f"mean={np.mean(noise_multipliers):.4f}",
                flush=True,
            )

        latest_global_model = None

        # ##DLG
        # start_idx = [0]
        # for i in range(1, num_clients):
        #     start_idx.append(start_idx[-1] + client_data_sizes[i - 1])
        # print(f"start_idx: {start_idx}")

        round_iter = trange(global_epoch) if args.verbose_logs else range(global_epoch)
        for epoch in round_iter:
            # precheck and pick up the candidates who can take the next commiunication round.
            candidates = [cid for cid in range(num_clients) if clients[cid].precheck()]
            if len(candidates) < get_selected_client_count():
                print('There are no enough clients can be trained!')
                break
            else:
                sampled_client_indices = select_clients(candidates, epoch + 1)
                if args.verbose_logs:
                    print(
                        f"round {epoch + 1}/{global_epoch}: sampled clients {sampled_client_indices}",
                        flush=True,
                    )
                sampled_clients_models = [clients_models[i] for i in sampled_client_indices]
                sampled_clients_train_loaders = [clients_train_loaders[i] for i in sampled_client_indices]
                sampled_clients_test_loaders = [clients_test_loaders[i] for i in sampled_client_indices]
                sampled_clients = [clients[i] for i in sampled_client_indices]
                # ##DLG
                # sampled_clients_idx = [start_idx[i] for i in sampled_client_indices]

                # download global model
                for client_model in sampled_clients_models:
                    client_model.load_state_dict(global_model.state_dict())
                clients_model_states = []
                clients_accuracies = []
                st_time = time.time()
                for idx, (client, client_model, client_trainloader, client_testloader) in enumerate(
                            zip(sampled_clients, sampled_clients_models, sampled_clients_train_loaders, sampled_clients_test_loaders)):
                    client_start_time = time.time()
                    if args.verbose_logs:
                        print(
                            f"round {epoch + 1}/{global_epoch}: "
                            f"client {idx + 1}/{len(sampled_clients)} "
                            f"cid={sampled_client_indices[idx]} "
                            f"data_size={client.data_size} start",
                            flush=True,
                        )
                    if latest_global_model is None:
                        local_update_first(model=client_model, dataloader=client_trainloader,
                                           global_model=global_model,
                                           client=client)
                    else:
                        local_update_decay(model=client_model, dataloader=client_trainloader,
                                           global_model=global_model,
                                           latest_global_model=latest_global_model,
                                           client=client)
                    # client_update = local_update_fedavg(model=client_model, dataloader=client_trainloader,
                    #                                     global_model=global_model,
                    #                                     client=client)
                    clients_model_states.append(clone_state_dict(client_model.state_dict()))
                    if args.eval_client_models:
                        accuracy = test(client_model, client_testloader)
                        clients_accuracies.append(accuracy)
                        accuracy_text = f" accuracy={accuracy:.4f}"
                    else:
                        accuracy_text = ""
                    if args.verbose_logs:
                        print(
                            f"round {epoch + 1}/{global_epoch}: "
                            f"client {idx + 1}/{len(sampled_clients)} "
                            f"cid={sampled_client_indices[idx]}"
                            f"{accuracy_text} "
                            f"elapsed={time.time() - client_start_time:.2f}s",
                            flush=True,
                        )
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
                if args.eval_client_models:
                    print(clients_accuracies)
                # ##DLG
                # if epoch == 19:
                #     dlg_attack(sampled_clients_models[0], dataset, epoch, idx=1)

                if args.eval_client_models:
                    mean_acc_s.append(sum(clients_accuracies) / len(clients_accuracies))
                    acc_matrix.append(clients_accuracies)
                sampled_client_data_sizes = [client_data_sizes[i] for i in sampled_client_indices]
                sampled_client_eps = [priv_preferences[i] for i in sampled_client_indices]

                aggregated_state = aggregate_client_states(
                    client_states=clients_model_states,
                    sampled_client_data_sizes=sampled_client_data_sizes,
                    sampled_client_eps=sampled_client_eps,
                    fedavg=fedavg,
                    weiavg=weiavg,
                    deavg=deavg,
                )
                global_model.load_state_dict(aggregated_state)
                en_time = time.time()
                global_accuracy, global_test_loss = evaluate(global_model, clients_test_loaders[0])
                if (epoch >= 2) and (global_accuracy >= global_acc[-1]) and (global_acc[-1] >= global_acc[-2]) and all(global_accuracy > x for x in global_acc):
                    latest_global_model = copy.deepcopy(global_model)
                    for client in clients:
                        client.ba.noise_multiplier *= decay_factor
                print(
                    'epoch:{}, global accuracy:{}, cost time:{:.2f}s'.format(
                        epoch + 1,
                        global_accuracy,
                        en_time - st_time,
                    ),
                    flush=True,
                )
                global_acc.append(global_accuracy)
                global_loss.append(global_test_loss)

        acc = pd.DataFrame({
            'round': list(range(1, len(global_acc) + 1)),
            'test_loss': global_loss,
            'test_accuracy': global_acc,
            'dataset': dataset,
            'method': get_method_name(),
            'client_num': num_clients,
            'privacy_setting': target_epsilon,
            'privacy_scenario': args.privacy_scenario,
            'iid': args.iid,
            'dir_alpha': args.dir_alpha,
            'client_fraction': user_sample_rate,
            'local_steps': local_epoch,
            'batch_size': batch_size,
            'lr': args.lr,
            'momentum': args.momentum,
            'weight_decay': args.weight_decay,
            'seed': args.seed,
            'phi': alpha,
            'fisher_threshold': args.fisher_threshold,
            'fisher_max_batches': args.fisher_max_batches,
            'fisher_estimator': args.fisher_estimator,
            'gamma': args.gamma,
            'max_clip_norm': args.max_clip_norm,
        })
        file_name = build_result_csv_path()
        acc.to_csv(file_name, index=False)
        print(f"Saved CSV result to: {file_name}")
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
