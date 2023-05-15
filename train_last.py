from torch.optim import lr_scheduler
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch import mps


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 卷積層
        self.conv1_layer = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1) # 第一层卷积
        torch.nn.init.xavier_normal(self.conv1_layer.weight)
        self.bn1_layer = nn.BatchNorm2d(64)     # BN 層
        self.conv2_layer = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1) # 第二层卷积
        torch.nn.init.xavier_normal(self.conv2_layer.weight)
        self.bn2_layer = nn.BatchNorm2d(128)    # BN 層
        self.conv3_layer = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1) # 第三层卷积
        torch.nn.init.xavier_normal(self.conv2_layer.weight)
        self.bn3_layer = nn.BatchNorm2d(256)    # BN 層
        self.relu_layer = nn.ReLU() # 激活函数
        self.maxpool_layer = nn.MaxPool2d(kernel_size=2, stride=2)  # 池化層

        self.dense_layer = nn.Sequential(      # 全连接层
            nn.Linear(28 * 28 * 256, 84),
            nn.BatchNorm1d(84),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(84, 500)
        )
        torch.nn.init.xavier_normal_(self.dense_layer[0].weight)
        torch.nn.init.xavier_normal_(self.dense_layer[4].weight)

    def forward(self, x):
        x = self.conv1_layer(x)
        x = self.bn1_layer(x)
        x = self.relu_layer(x)
        x = self.maxpool_layer(x)

        x = self.conv2_layer(x)
        x = self.bn2_layer(x)
        x = self.relu_layer(x)
        x = self.maxpool_layer(x)

        x = self.conv3_layer(x)
        x = self.bn3_layer(x)
        x = self.relu_layer(x)
        x = self.maxpool_layer(x)

        x = x.view(-1, 28 * 28 * 256)
        x = self.dense_layer(x)

        return x


if __name__ == '__main__':

    device = torch.device('mps')    # 設置為GPU運行

    # 参数设置
    n_epochs = 60
    num_classes = 10
    learning_rate = 0.0001
    momentum = 0.9
    batchSize = 64

    # ============================ step 1/5 数据 ============================

    # normalize = transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])  # 规范化
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.RandomCrop([224, 224]),
        transforms.ToTensor(),
        # transforms.RandomHorizontalFlip(p=5),
        # transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
        transforms.ColorJitter(0.4, 0.4, 0.4),  # 随机颜色变换
        # transforms.Grayscale(num_output_channels=3),   # 黑白照
        # transforms.RandomRotation(30),  # 随机旋转
        # # transforms.Normalize([0.485, 0.456, 0.406],  # 对图像像素进行归一化
        # #                      [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=1, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=(254/255, 0, 0)),
        transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]),  # 规范化
        transforms.RandomAffine(30), # 再改成50
        # transforms.RandomOrder([transforms.RandomRotation(15),
        #                         transforms.Pad(padding=32),
        #                         transforms.RandomAffine(degrees=0, translate=(0.01, 0.1), scale=(0.9, 1.1))]),

        # transforms.RandomRotation(30),
    ])

    test_transform = transforms.Compose([  # 数据处理
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # trainset = torchvision.datasets.CIFAR10(root='./Cifar-10',
    #                                         train=True, download=True, transform=data_transform)

    trainset = ImageFolder('./train_sample', data_transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize, shuffle=True) # 构建DataLoder

    testset = ImageFolder('./dev_sample', test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize, shuffle=False)  # 构建DataLoder

    # model = CNN().to(device)

# ============================ step 2/5 模型 ============================

    model = CNN().to(device)
    model.load_state_dict(torch.load("model_parameter3.pkl"), device)

# ============================ step 3/5 损失函数 ============================

    criterion = torch.nn.CrossEntropyLoss() # 设置学习率下降策略

# ============================ step 4/5 优化器 ============================
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 选择优化器
    lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # 设置学习率下降策略

    best_acc = 37

# ============================ step 5/5 训练 ============================
    train_curve = list()    # 用於收集train loss 變化，方便後面畫圖
    test_curve = list()

    for epoch in range(n_epochs):
        mps.empty_cache()

        print("Epoch {}/{}".format(epoch, n_epochs))
        print("-" * 10)

        running_loss = 0.0
        running_correct = 0
        model.train()
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

        model.eval()
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


torch.save(model.state_dict(), "model_parameter3.pkl")

