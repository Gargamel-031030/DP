import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


def weights_init(m):
    if hasattr(m, "weight"):
        m.weight.data.uniform_(-0.5, 0.5)
    if hasattr(m, "bias"):
        m.bias.data.uniform_(-0.5, 0.5)


class mnistNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 8, 2, padding=3)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        self.fc1 = nn.Linear(32 * 4 * 4, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        # x of shape [B, 1, 28, 28]
        x = F.relu(self.conv1(x))  # -> [B, 16, 14, 14]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 16, 13, 13]
        x = F.relu(self.conv2(x))  # -> [B, 32, 5, 5]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 32, 4, 4]
        x = x.view(-1, 32 * 4 * 4)  # -> [B, 512]
        x = F.relu(self.fc1(x))  # -> [B, 32]
        x = self.fc2(x)  # -> [B, 10]
        output = F.log_softmax(x, dim=1)
        return output

    def name(self):
        return "mnistNet"


class fmnistNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 8, 2, padding=3)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        self.fc1 = nn.Linear(32 * 4 * 4, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        # x of shape [B, 1, 28, 28]
        x = F.relu(self.conv1(x))  # -> [B, 16, 14, 14]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 16, 13, 13]
        x = F.relu(self.conv2(x))  # -> [B, 32, 5, 5]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 32, 4, 4]
        x = x.view(-1, 32 * 4 * 4)  # -> [B, 512]
        x = F.relu(self.fc1(x))  # -> [B, 32]
        x = self.fc2(x)  # -> [B, 10]
        output = F.log_softmax(x, dim=1)
        return output

    def name(self):
        return "fmnistNet"


class cifarNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(128 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

    def name(self):
        return "cifarNet"


def cifar100ResNet18():
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 100)
    return model


'''
自定义LeNet网络
'''
class LeNet(nn.Module): # nn.Module, 定义神经网络必须继承的模块， 框架规定的形式
    def __init__(self, channel=3, hidden=768, num_classes=10): # 假设输入cifar10数据集， 默认3通道， 隐层维度为768， 分类为10
        super(LeNet, self).__init__() # 继承pytorch神经网络工具箱中的模块
        act = nn.Sigmoid # 激活函数为Sigmoid
        # nn.Sequential: 顺序容器。 模块将按照在构造函数中传递的顺序添加到模块中。 或者，也可以传递模块的有序字典
        self.body = nn.Sequential( # 设计神经网络结构，对于nn.Sequential.Preference : https://zhuanlan.zhihu.com/p/75206669
            # 设计输入通道为channel，输出通道为12， 5x5卷积核尺寸，填充为5 // 2是整除。故填充为2， 步长为2的卷积层
            nn.Conv2d(channel, 12, kernel_size=5, padding=5 // 2, stride=2),
            # 经过卷积后， 使用Sigmoid激活函数激活
            act(),
            # 设计输入通道为12，输出通道为12， 5x5卷积核尺寸，填充为5 // 2是整除。故填充为2， 步长为2的卷积层
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            # 经过卷积后， 使用Sigmoid激活函数激活
            act(),
            # 设计输入通道为12，输出通道为12， 5x5卷积核尺寸，填充为5 // 2是整除。故填充为2， 步长为1的卷积层
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            # 经过卷积后， 使用Sigmoid激活函数激活
            act()
        )
        # 设计一个全连接映射层， 将hidden隐藏层映射到十个分类标签
        self.fc = nn.Sequential(
            nn.Linear(hidden, num_classes)
        )

    # 设计前向传播算法
    def forward(self, x):
        out = self.body(x) # 先经过nn.Sequential的顺序层得到一个输出
        out = out.view(out.size(0), -1) # 将输出转换对应的维度
        out = self.fc(out) # 最后将输出映射到一个十分类的一个列向量
        return out
