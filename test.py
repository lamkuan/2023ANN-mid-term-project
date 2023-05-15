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
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1) # 第一层卷积
        torch.nn.init.xavier_normal(self.conv1.weight)
        self.bn1 = nn.BatchNorm2d(64)     # BN 層
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1) # 第二层卷积
        torch.nn.init.xavier_normal(self.conv2.weight)
        self.bn2 = nn.BatchNorm2d(128)    # BN 層
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1) # 第三层卷积
        torch.nn.init.xavier_normal(self.conv2.weight)
        self.bn3 = nn.BatchNorm2d(256)    # BN 層
        self.relu = nn.ReLU() # 激活函数
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)  # 池化層

        self.dense = nn.Sequential(      # 全连接层
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

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = x.view(-1, 28 * 28 * 256)
        x = self.dense(x)

        return x


if __name__ == '__main__':

    device = torch.device('mps')    # 設置為GPU運行

    test_transform = transforms.Compose([  # 数据处理
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    batchSize = 64

    testset = ImageFolder('./test_sample', test_transform)

    testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize, shuffle=False)  # 构建DataLoder

    model = CNN().to(device)
    model.load_state_dict(torch.load("model_parameter2.pkl"), device)

    testing_correct = 0

    model.eval()

    for data in testloader:
        X_test, y_test = data
        X_test, y_test = X_test.to(device), y_test.to(device)
        outputs = model(X_test)
        _, pred = torch.max(outputs.data, 1)
        testing_correct += torch.sum(pred == y_test.data)

    test_acc = torch.true_divide(100 * testing_correct, len(testset))

    print("Test Accuracy is:{:.4f}".format(test_acc))

    # if test_acc > best_acc:
    #     print("model update")
    #     torch.save(model.state_dict(), "model_parameter3.pkl")
    #     best_acc = test_acc