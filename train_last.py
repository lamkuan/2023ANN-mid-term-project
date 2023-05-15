import torch
import torch.nn as nn
import torch.functional as F
import torchvision
import torchvision.transforms as transforms
import tqdm
from torch.optim import lr_scheduler
# from torchsummary import summary
import tensorboard
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
import gc


class VGG16(nn.Module):
    def __init__(self, in_channels=1, num_classes=1000):
        super(VGG16, self).__init__()

        # block_1
        self.c1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.a1 = nn.ReLU(inplace=True)

        self.c2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.a2 = nn.ReLU(inplace=True)
        self.p2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # block_2
        self.c3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.a3 = nn.ReLU(inplace=True)

        self.c4 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.a4 = nn.ReLU(inplace=True)
        self.p4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # block_3
        self.c5 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.a5 = nn.ReLU(inplace=True)

        self.c6 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.a6 = nn.ReLU(inplace=True)

        self.c7 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.a7 = nn.ReLU(inplace=True)
        self.p7 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # block_4
        self.c8 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.a8 = nn.ReLU(inplace=True)

        self.c9 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.a9 = nn.ReLU(inplace=True)

        self.c10 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.a10 = nn.ReLU(inplace=True)
        self.p10 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # block_5
        self.c11 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.a11 = nn.ReLU(inplace=True)

        self.c12 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.a12 = nn.ReLU(inplace=True)

        self.c13 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.a13 = nn.ReLU(inplace=True)
        self.p13 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # self.fc1_d=nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.fc1_a = nn.ReLU(inplace=True)

        # self.fc2_d=nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc2_a = nn.ReLU(inplace=True)

        self.fc3 = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.c1(x)
        x = self.a1(x)
        x = self.c2(x)
        x = self.a2(x)
        x = self.p2(x)

        x = self.c3(x)
        x = self.a3(x)
        x = self.c4(x)
        x = self.a4(x)
        x = self.p4(x)

        x = self.c5(x)
        x = self.a5(x)
        x = self.c6(x)
        x = self.a6(x)
        x = self.c7(x)
        x = self.a7(x)
        x = self.p7(x)

        x = self.c8(x)
        x = self.a8(x)
        x = self.c9(x)
        x = self.a9(x)
        x = self.c10(x)
        x = self.a10(x)
        x = self.p10(x)

        x = self.c11(x)
        x = self.a11(x)
        x = self.c12(x)
        x = self.a12(x)
        x = self.c13(x)
        x = self.a13(x)
        x = self.p13(x)

        x = torch.flatten(x, start_dim=1)

        x = self.fc1(x)
        x = self.fc1_a(x)

        x = self.fc2(x)
        x = self.fc2_a(x)

        x = self.fc3(x)
        return x


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.channels = channels

        self.conv1 = torch.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        torch.nn.init.kaiming_normal(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        self.bn1 = torch.nn.BatchNorm2d(channels)

        self.conv2 = torch.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        torch.nn.init.kaiming_normal(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        self.bn2 = torch.nn.BatchNorm2d(channels)

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = nn.ReLU()(y)
        y = self.conv2(y)
        y = self.bn2(y)

        return nn.ReLU()(x + y)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        torch.nn.init.xavier_normal(self.conv1.weight)
        self.bn1 = nn.BatchNorm2d(64)
        # self.res1 = ResidualBlock(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        torch.nn.init.xavier_normal(self.conv2.weight)
        self.bn2 = nn.BatchNorm2d(128)
        # self.res2 = ResidualBlock(128)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        torch.nn.init.xavier_normal(self.conv2.weight)
        self.bn3 = nn.BatchNorm2d(256)
        # self.res3 = ResidualBlock(256)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dense = nn.Sequential(
            nn.Linear(28 * 28 * 256, 84),
            nn.BatchNorm1d(84),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(84, 500)
        )
        torch.nn.init.xavier_normal_(self.dense[0].weight)
        torch.nn.init.xavier_normal_(self.dense[4].weight)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # x = self.res1(x)


        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # x = self.res2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # x = self.res3(x)

        # x = x.view(x.size(0), -1)
        x = x.view(-1, 28 * 28 * 256)
        x = self.dense(x)

        return x


if __name__ == '__main__':

    device = torch.device('mps')

    batchSize = 64
    # normalize = transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])  # 规范化
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.RandomCrop([224, 224]),
        transforms.ToTensor(),
        # transforms.RandomHorizontalFlip(p=5),
        # transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
        # transforms.ColorJitter(0.4, 0.4, 0.4),  # 随机颜色变换
        # transforms.Grayscale(num_output_channels=3),   # 黑白照
        # transforms.RandomRotation(30),  # 随机旋转
        # transforms.Normalize([0.485, 0.456, 0.406],  # 对图像像素进行归一化
        #                      [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=1, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=(254/255, 0, 0)),
        transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]),  # 规范化
        # transforms.RandomAffine(30), # 再改成50
        transforms.RandomOrder([transforms.RandomRotation(15),
                                transforms.Pad(padding=32),
                                transforms.RandomAffine(degrees=0, translate=(0.01, 0.1), scale=(0.9, 1.1))]),

        # transforms.RandomRotation(30),
        # normalize()
    ])

    test_transform = transforms.Compose([  # 数据处理
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.RandomCrop(100),
        # transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(brightness=1),

    ])

    # trainset = torchvision.datasets.CIFAR10(root='./Cifar-10',
    #                                         train=True, download=True, transform=data_transform)

    trainset = ImageFolder('./train_sample', data_transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize, shuffle=True)

    testset = ImageFolder('./dev_sample', test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize, shuffle=False)

    # model = VGG16(in_channels=3, num_classes=10).to(device)

    # model = CNN().to(device)
    #
    model = CNN().to(device)
    # model = torch.load_state_dict("model_parameter2.pkl", device)
    model.load_state_dict(torch.load("model_parameter3.pkl"), device)
    model.eval()
    # model = model.state_dict()

    n_epochs = 60
    num_classes = 10
    learning_rate = 0.0001
    momentum = 0.9

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_acc = -1

    for epoch in range(n_epochs):
        mps.empty_cache()

        print("Epoch {}/{}".format(epoch, n_epochs))
        print("-" * 10)

        running_loss = 0.0
        running_correct = 0
        for data in trainloader:
            X_train, y_train = data
            X_train, y_train = X_train.to(device), y_train.to(device)
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            _, pred = torch.max(outputs.data, 1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.data.item()
            running_correct += torch.sum(pred == y_train.data)

        testing_correct = 0

        for data in testloader:
            X_test, y_test = data
            X_test, y_test = X_test.to(device), y_test.to(device)
            outputs = model(X_test)
            _, pred = torch.max(outputs.data, 1)
            testing_correct += torch.sum(pred == y_test.data)

        test_acc = torch.true_divide(100 * testing_correct, len(testset))

        print("Loss is:{:.4f}, Train Accuracy is:{:.4f}%, Test Accuracy is:{:.4f}".format(
            torch.true_divide(running_loss, len(trainset)),
            torch.true_divide(100 * running_correct, len(trainset)),
            test_acc))

        if test_acc > best_acc:
            print("model update")
            torch.save(model.state_dict(), "model_parameter3.pkl")
            best_acc = test_acc

