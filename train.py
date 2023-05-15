import os
import gc
import time
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch import mps
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T
import torch.nn.functional as F
from PIL import Image

batch_size = 128


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
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

        # self.out = nn.Linear(16 * 53 * 53, 120)
        self.out = nn.Linear(7 * 7 * 128, 120)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)

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
            #torch.nn.Softmax(dim=1)
            #dim=1是按行softmax——降到（0,1）区间内相当于概率，此处不用softmax因为定义的交叉熵损失函数CrossEntropy包含了softmax
        )

    def forward(self, x):
        x = self.conv(x)
        #print(x.size())
        x = x.contiguous().view(-1,256*6*6)  #使用.contiguous()防止用多卡训练的时候tensor不连续，即tensor分布在不同的内存或显存中
        x = self.fc(x)
        return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=6,
                kernel_size=5,
                # stride=1,
                # padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=6,
                out_channels=16,
                kernel_size=5,
                # stride=1,
                # padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.out = nn.Sequential(
            nn.Linear(16 * 53 * 53, 120),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)

        return output



def accuracy(predictions, labels):
    pred = torch.max(predictions.data, 1)[1]
    rights = pred.eq(labels.data.view_as(pred)).sum()
    return rights, len(labels)


if __name__ == "__main__":

    device = torch.device("mps" if torch.backends.mps.is_available else "cpu")  # 初始化為device

    total_accuracy = 0
    best_acc = 0

    print("是否使用GPU训练：{}".format(torch.backends.mps.is_available()))  # 打印是否采用gpu训练

    # if torch.backends.mps.is_available():
    #     device = torch.device("mps");  # 打印相应的gpu信息

    normalize = transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])  # 规范化
    transform = transforms.Compose([  # 数据处理
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        normalize
    ])

    dataset_train = ImageFolder('./train_sample', transform=transform)  # 训练数据集
    # print(dataset_tran[0])
    dataset_valid = ImageFolder('./test_sample', transform=transform)  # 验证或测试数据集

    # dataset_train = MyDataset('./train_sample', train=True)
    # dataset_valid = MyDataset('./test_sample', train=True)

    # print(dataset_train.classer)#返回类别
    print(dataset_train.class_to_idx)  # 返回类别及其索引

    print(dataset_valid.class_to_idx)

    train_data_size = len(dataset_train)  # 放回数据集长度
    test_data_size = len(dataset_valid)

    print("训练数据集的长度为：{}".format(train_data_size))
    print("测试数据集的长度为：{}".format(test_data_size))

    sampler_train = RandomSampler(dataset_train)
    sampler_test = RandomSampler(dataset_valid)

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, num_workers=0, drop_last=True, sampler=sampler_train)
    dataloader_test = DataLoader(dataset_valid, batch_size=batch_size, num_workers=0, drop_last=True, sampler=sampler_test)

    # 2.模型加载
    model_ft = Net()

    # print(model_ft)  # 打印模型資料

    # model_ft = model_ft.to(device)  # 将模型迁移到gpu
    model_ft = model_ft.float().to(device)

    loss_fn = nn.CrossEntropyLoss()

    loss_fn = loss_fn.float().to(device)  # 将loss迁移到gpu
    # loss_fn = loss_fn.float()

    learn_rate = 0.01  # 设置学习率
    optimizer = torch.optim.SGD(model_ft.parameters(), lr=learn_rate, momentum=0.01)  # 可调超参数

    total_train_step = 0
    total_test_step = 0
    epoch = 1000  # 迭代次数
    # writer = SummaryWriter("logs_train_yaopian")
    best_acc = -1
    ss_time = time.time()

    for i in range(epoch):
        mps.empty_cache()
        start_time = time.time()
        train_rights = []
        print("--------第{}轮训练开始---------".format(i + 1))
        model_ft.train()
        for index, data in enumerate(dataloader_train):
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = model_ft(imgs).to(device)
            loss = loss_fn(outputs, targets).to(device)

            right = accuracy(outputs, targets)
            train_rights.append(right)

            optimizer.zero_grad()  # 梯度归零
            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 梯度优化

            if index % 100 == 0:  # 一轮时间过长可以考虑加一个
                model_ft.eval()
                end_time = time.time()

                val_rights = []

                with torch.no_grad():  # 验证数据集时禁止反向传播优化权重
                    for data2 in dataloader_test:
                        imgs2, targets2 = data2

                        imgs2 = imgs2.to(device)
                        targets2 = targets2.to(device)

                        outputs2 = model_ft(imgs2)
                        loss2 = loss_fn(outputs2, targets2)

                        right = accuracy(outputs2, targets2)
                        val_rights.append(right)

                    train_r = (sum(tup[0] for tup in train_rights), sum(tup[1] for tup in train_rights))
                    val_r = (sum(tup[0] for tup in val_rights), sum(tup[1] for tup in val_rights))

                    total_accuracy = 100. * val_r[0].cpu().numpy() / val_r[1]

                        # del data2
                        # del imgs2
                        # del targets2
                        # del outputs2
                        # del loss2
                        #
                        # gc.collect()

                        # if i % 100 == 0:
                    print('當前epoch: {} [{}/{} ({:.0f}%)]\t損失: {:.6f}\t訓練習準確率: {:.2f}%\t測試習準確率: {:.2f}'.format(
                        i, index * batch_size, len(dataloader_train.dataset), 100. * i / len(dataloader_train),
                        loss.data, 100. * train_r[0].cpu().numpy() / train_r[1], 100. * val_r[0].cpu().numpy() / val_r[1]))

                    if total_accuracy > best_acc:  # 保存迭代次数中最好的模型
                        print("已修改模型")
                        best_acc = total_accuracy
                        torch.save(model_ft, "model2.pth")

            # del data
            # del imgs
            # del targets
            # del outputs
            # del loss
            #
            # gc.collect()

