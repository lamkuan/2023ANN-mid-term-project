import os
import gc
import time
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T
import torch.nn.functional as F
from PIL import Image

batch_size = 64


class MyDataset(Dataset):
    def __init__(self, root, transforms=None, train=True):
        imgs = [os.path.join(root, img) for img in os.listdir(root)]

        imgs_num = len(imgs)

        if train:
            self.imgs = imgs[:int(0.7 * imgs_num)]
        else:
            self.imgs = imgs[int(0.3 * imgs_num):]

        if transforms is None:
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

            self.transforms = T.Compose([
                T.Resize(224),
                T.CenterCrop(224),
                T.ToTensor(),
                normalize
            ])
        else:
            self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.imgs[index]
        # dog label : 1           cat label : 0
        label = 1 if "dog" in img_path.split('/')[-1] else 0
        data = Image.open(img_path)
        if data.mode != 'L':
            data = data.convert('L')

        data = self.transforms(data)
        return data, label

    def __len__(self):
        return len(self.imgs)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
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
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 128, 5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # self.out = nn.Linear(16 * 53 * 53, 120)
        self.out = nn.Linear(128 * 3 * 3, 120)

        print(self.out)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)

        return output


class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(16 * 53 * 53, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 500)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 将x通过第一个卷积层和ReLU激活函数，然后通过最大池化层
        x = self.pool(F.relu(self.conv2(x)))  # 将x通过第二个卷积层和ReLU激活函数，然后通过最大池化层
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 16 * 53 * 53)  # 将x展平为一维向量，-1表示自动推断批量大小
        x = F.relu(self.fc1(x))  # 将x通过第一个全连接层和ReLU激活函数
        x = F.relu(self.fc2(x))  # 将x通过第二个全连接层和ReLU激活函数
        x = self.fc3(x)  # 将x通过第三个全连接层
        x = F.softmax(x, dim=1)  # 将x通过softmax函数，沿着第一个维度（类别维度）进行归一化，得到类别概率分布
        return x  # 返回输出的类别概率分布


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv4 = nn.Conv2d(128, 128, 3)
        self.max_pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        # 12*12 for size(224,224)    7*7 for size(150,150)
        self.fc1 = nn.Linear(128 * 12 * 12, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        in_size = x.size(0)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.max_pool(x)
        # 展开
        x = x.view(in_size, -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


if __name__ == "__main__":

    device = "cpu"  # 初始化為CPU

    print("是否使用GPU训练：{}".format(torch.backends.mps.is_available()))  # 打印是否采用gpu训练

    if torch.backends.mps.is_available():
        device = torch.device("mps");  # 打印相应的gpu信息

    normalize = transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])  # 规范化
    transform = transforms.Compose([  # 数据处理
        transforms.Resize((224, 224)),
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

    dataloader_train = DataLoader(dataset_train, batch_size=64, num_workers=0, drop_last=True, sampler=sampler_train)
    dataloader_test = DataLoader(dataset_valid, batch_size=64, num_workers=0, drop_last=True, sampler=sampler_test)

    # 2.模型加载
    model_ft = CNN()

    print(model_ft)  # 打印模型資料

    model_ft = model_ft.to(device)  # 将模型迁移到gpu

    loss_fn = nn.CrossEntropyLoss()

    loss_fn = loss_fn.to(device)  # 将loss迁移到gpu

    learn_rate = 0.01  # 设置学习率
    optimizer = torch.optim.SGD(model_ft.parameters(), lr=learn_rate, momentum=0.01)  # 可调超参数

    total_train_step = 0
    total_test_step = 0
    epoch = 50  # 迭代次数
    writer = SummaryWriter("logs_train_yaopian")
    best_acc = -1
    ss_time = time.time()

    for i in range(epoch):
        torch.mps.empty_cache()
        start_time = time.time()
        print("--------第{}轮训练开始---------".format(i + 1))
        model_ft.train()
        for data in dataloader_train:
            imgs, targets = data
            # if torch.cuda.is_available():
            # imgs.float()
            # imgs=imgs.float()#为上述改为半精度操作，在这里不适用
            #         imgs=imgs.cuda()
            imgs = imgs.to(device)
            #         targets=targets.cuda()
            targets = targets.to(device)
            # imgs=imgs.half()
            outputs = model_ft(imgs).to(device)
            loss = loss_fn(outputs, targets)

            optimizer.zero_grad()  # 梯度归零
            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 梯度优化

            total_train_step = total_train_step + 1
            if total_train_step % 100 == 0:  # 一轮时间过长可以考虑加一个
                end_time = time.time()
                print("使用GPU训练100次的时间为：{}".format(end_time - start_time))
                print("训练次数：{},loss:{}".format(total_train_step, loss.item()))
                # writer.add_scalar("valid_loss",loss.item(),total_train_step)

            del imgs
            del targets
            del outputs
            del loss

            gc.collect()

        model_ft.eval()
        total_test_loss = 0
        total_accuracy = 0

        with torch.no_grad():  # 验证数据集时禁止反向传播优化权重
            for data in dataloader_test:
                imgs, targets = data
                # if torch.cuda.is_available():
                # imgs.float()
                # imgs=imgs.float()
                imgs = imgs.to(device)
                targets = targets.to(device)
                # imgs=imgs.half()
                outputs = model_ft(imgs)
                loss = loss_fn(outputs, targets)
                total_test_loss = total_test_loss + loss.item()
                accuracy = (outputs.argmax(1) == targets).sum()
                total_accuracy = total_accuracy + accuracy
            print("整体测试集上的loss：{}(越小越好,与上面的loss无关此为测试集的总loss)".format(total_test_loss))
            print("整体测试集上的正确率：{}(越大越好)".format(total_accuracy / len(dataset_valid)))

            writer.add_scalar("valid_loss", (total_accuracy / len(dataset_valid)), (i + 1))  # 选择性使用哪一个
            total_test_step = total_test_step + 1
            if total_accuracy > best_acc:  # 保存迭代次数中最好的模型
                print("已修改模型")
                best_acc = total_accuracy
                torch.save(model_ft, "model.pth")

        ee_time = time.time()
        zong_time = ee_time - ss_time
        print("训练总共用时:{}h:{}m:{}s".format(int(zong_time // 3600), int((zong_time % 3600) // 60),
                                          int(zong_time % 60)))  # 打印训练总耗时
        writer.close()
