import torch.random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset, SubsetRandomSampler
from fedlab.utils.dataset.functional import hetero_dir_partition
from typing import Tuple, List
from options import parse_args
import numpy as np

args = parse_args()

torch.manual_seed(0)
torch.cuda.manual_seed(0)
def get_client_example_nums(num_examples_per_client, num_clients, num_examples):
    # 设置随机数种子以便结果可复现
    np.random.seed(0)
    # 生成19个范围在2900到3100之间的整数随机数
    client_example_nums = np.random.randint(num_examples_per_client-200, num_examples_per_client+201, num_clients-1)  # 注意range是闭开区间，所以3101不会被选中
    # 计算第20个整数，使得总和为60000
    total_sum = np.sum(client_example_nums)
    last_number = num_examples - total_sum
    # 确保最后一个数也是整数（由于前面19个数的和可能非常接近60000，这通常不是问题）
    # 但如果需要强制整数，可以采取四舍五入的方式（如果误差在可接受范围内）
    last_number = round(last_number)
    # 将第20个整数添加到列表中
    client_example_nums = np.append(client_example_nums, last_number)
    print("每个client的样本数：")
    print(client_example_nums)
    print("所有client样本数的和：", np.sum(client_example_nums))
    return client_example_nums

def get_clients_datasets(train_dataset, num_clients):

    n = len(train_dataset)
    indices = list(range(n))
    split_size = n // num_clients
    client_example_nums = get_client_example_nums(num_examples_per_client=split_size, num_clients=num_clients,
                                                  num_examples=n)
    clients_datasets = []
    last_index = 0
    for cid in range(num_clients):
        client_indices = indices[last_index: last_index+client_example_nums[cid]]
        client_dataset = Subset(train_dataset, client_indices)
        clients_datasets.append(client_dataset)
        last_index = client_example_nums[cid]

    return clients_datasets

#MNIST-------------------------------------------------------------------------------------------------------
def get_mnist_datasets():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    train_dataset = datasets.MNIST('./data/MNIST', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data/MNIST', train=False, download=True, transform=transform)

    return train_dataset, test_dataset

#fMNIST-------------------------------------------------------------------------------------------------------
def get_fmnist_datasets():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    train_dataset = datasets.FashionMNIST('./data/FMNIST', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST('./data/FMNIST', train=False, download=True, transform=transform)

    return train_dataset, test_dataset

def get_noniid_fmnist(alpha: float, num_clients: int) -> Tuple[List[DataLoader], List[DataLoader], List[int]]:
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    train_dataset = datasets.FashionMNIST('./data/FMNIST', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST('./data/FMNIST', train=False, download=True, transform=transform)

    num_classes = len(np.unique(train_dataset.targets))

    train_partition = hetero_dir_partition(train_dataset.targets, num_clients, num_classes, alpha)

    train_loaders = []
    test_loaders = []
    client_data_sizes = []

    # Create a shared test_loader for all clients
    shared_test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True)

    for i in range(num_clients):
        train_sampler = torch.utils.data.SubsetRandomSampler(train_partition[i])

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, drop_last=True)

        train_loaders.append(train_loader)
        test_loaders.append(shared_test_loader)
        client_data_sizes.append(len(train_partition[i]))

        # Calculate and print label percentages for each client
        label_counts = np.zeros(num_classes)
        for idx in train_partition[i]:
            label_counts[train_dataset.targets[idx]] += 1
        label_percentages = label_counts / len(train_partition[i]) * 100

        # print(f"Client {i}: Label Percentages:")
        # for label, percentage in enumerate(label_percentages):
        #     print(f"Label {label}: {percentage:.2f}%")

    return train_loaders, test_loaders, client_data_sizes

#CIFAR10-------------------------------------------------------------------------------------------------------
def get_CIFAR10(alpha: float, num_clients: int) -> Tuple[List[DataLoader], List[DataLoader], List[int]]:
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = datasets.CIFAR10(root='./data/CIFAR10', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data/CIFAR10', train=False, download=True, transform=transform)

    num_classes = len(np.unique(train_dataset.targets))

    train_partition = hetero_dir_partition(train_dataset.targets, num_clients, num_classes, alpha)

    train_loaders = []
    test_loaders = []
    client_data_sizes = []

    # Create a shared test_loader for all clients
    shared_test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True)

    for i in range(num_clients):
        train_sampler = torch.utils.data.SubsetRandomSampler(train_partition[i])

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, drop_last=True)

        train_loaders.append(train_loader)
        test_loaders.append(shared_test_loader)
        client_data_sizes.append(len(train_partition[i]))

        # Calculate and print label percentages for each client
        label_counts = np.zeros(num_classes)
        for idx in train_partition[i]:
            label_counts[train_dataset.targets[idx]] += 1
        label_percentages = label_counts / len(train_partition[i]) * 100

        # print(f"Client {i}: Label Percentages:")
        # for label, percentage in enumerate(label_percentages):
        #     print(f"Label {label}: {percentage:.2f}%")

    return train_loaders, test_loaders, client_data_sizes