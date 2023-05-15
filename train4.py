import time
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch import mps, optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T
import torch.nn.functional as F
from PIL import Image

input_size = 224
num_classes = 10
num_epochs = 60
batch_size = 128

# normalize = transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])  # 规范化
normalize = transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])  # 规范化
transform = transforms.Compose([  # 数据处理
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.RandomCrop(100),
    # transforms.RandomHorizontalFlip(),
    # transforms.ColorJitter(brightness=1),
    normalize
])

# train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)

train_dataset = ImageFolder('./train_sample', transform)

# test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

test_dataset = ImageFolder('./test_sample', transform)

sampler_train = RandomSampler(train_dataset)
sampler_test = RandomSampler(test_dataset)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, num_workers=0, drop_last=True, batch_size=batch_size, sampler=sampler_train)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, num_workers=0, drop_last=True, batch_size=batch_size, sampler=sampler_test)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)

        return output


class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=6,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 128, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 256, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(256, 512, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # self.out = nn.Linear(16 * 53 * 53, 120)
        # self.out = nn.Linear(7 * 7 * 128, 120)
        self.fc1 = nn.Linear(1 * 1 * 512, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 500)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = F.softmax(x, dim=1)

        return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=6,
                kernel_size=5,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(6),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=6,
                out_channels=16,
                kernel_size=5,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(16),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(32),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=5,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(64),
        )

        # self.conv5 = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels=32,
        #         out_channels=64,
        #         kernel_size=5,
        #     ),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2),
        #     nn.BatchNorm2d(64),
        # )

        self.out = nn.Sequential(
            nn.Linear(64 * 3 * 3, 120),
            nn.ReLU(),
            # nn.BatchNorm1d(120),
        )

        self.fc2 = self.out = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(),
            # nn.BatchNorm1d(120),
        )

        self.fc3 = self.out = nn.Sequential(
            nn.Linear(84, 500),
            nn.ReLU(),
            # nn.BatchNorm1d(120),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        x = self.fc2(x)
        x = self.fc3(x)
        output = F.softmax()
        return output


# 定义网络结构
class AlexNet(torch.nn.Module):
    def __init__(self):
        super(AlexNet,self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3,
                            out_channels=96,
                            kernel_size=11,
                            stride=4),
            torch.nn.BatchNorm2d(96),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3,2),
            torch.nn.Conv2d(96,256,5,padding=2),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3,2),
            torch.nn.Conv2d(256,384,3,padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(384,384,3,padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(384,256,3,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3,2),
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(256*6*6,4096),
            torch.nn.ReLU(),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(),
            torch.nn.Linear(4096, 2),
            torch.nn.Softmax(dim=1)
            #dim=1是按行softmax——降到（0,1）区间内相当于概率，此处不用softmax因为定义的交叉熵损失函数CrossEntropy包含了softmax
        )

    def forward(self, x):
        x = self.conv(x)
        #print(x.size())
        x = x.contiguous().view(-1,256*6*6)  #使用.contiguous()防止用多卡训练的时候tensor不连续，即tensor分布在不同的内存或显存中
        x = self.fc(x)
        return x


def accuracy(predictions, labels):
    pred = torch.max(predictions.data, 1)[1]
    rights = pred.eq(labels.data.view_as(pred)).sum()
    return rights, len(labels)


if __name__ == "__main__":

    device = torch.device("mps" if torch.backends.mps.is_available else "cpu")  # 初始化為device

    net = torch.load("model5.pth", device)

    # net = Net().float().to(device)

    criterion = nn.CrossEntropyLoss().float().to(device)
    learn_rate = 0.1  # 设置学习率
    optimizer = torch.optim.SGD(net.parameters(), lr=learn_rate, momentum=0.01)  # 可调超参数

    best_acc = 14.58
    # best_acc = -1
    total_accuracy = 0

    for epoch in range(num_epochs):
        mps.empty_cache()
        train_rights = []

        for batch_idx, (data, target) in enumerate(train_loader):
            net.train()

            target = target.to(device)
            data = data.to(device)

            output = net(data)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            right = accuracy(output, target)
            train_rights.append(right)

            if batch_idx % 100 == 0:
                net.eval()
                val_rights = []

                with torch.no_grad():  # 验证数据集时禁止反向传播优化权重
                    for (data, target) in test_loader:
                        data = data.to(device)
                        target = target.to(device)
                        output = net(data)
                        right = accuracy(output, target)
                        val_rights.append(right)

                    train_r = (sum(tup[0] for tup in train_rights), sum(tup[1] for tup in train_rights))
                    val_r = (sum(tup[0] for tup in val_rights), sum(tup[1] for tup in val_rights))

                    # total_accuracy = 100. * val_r[0].cpu().numpy() / val_r[1]

                    total_accuracy = 100. * val_r[0].cpu().numpy() / val_r[1]

                    print('當前epoch: {} [{}/{} ({:.0f}%)]\t損失: {:.6f}\t訓練習準確率: {:.2f}%\t測試習準確率: {:.2f}'.format(
                        epoch, batch_idx * batch_size, len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                        loss.data, 100. * train_r[0].cpu().numpy() / train_r[1], 100. * val_r[0].cpu().numpy() / val_r[1]))

                    if best_acc < total_accuracy:
                        print("已修改模型")
                        best_acc = total_accuracy
                        torch.save(net, "model5.pth")

