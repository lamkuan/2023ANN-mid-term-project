import torch.nn as nn


class MyNet(nn.Module):
    # 可以根据需要调整网络层数和每层的卷积核大小、通道数等参数
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv5 = nn.Conv2d(256, 512, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(512 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, 2)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = self.relu(self.conv4(x))
        x = self.pool(x)
        x = self.relu(self.conv5(x))
        x = self.pool(x)
        x = x.view(-1, 512 * 4 * 4)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 26 * 26, 512)
        # self.fc2 = nn.Linear(512, 256)
        # self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.conv3(x))
        x = nn.functional.max_pool2d(x, 2)
        # x = nn.functional.relu(self.conv4(x))
        # x = nn.functional.max_pool2d(x, 2)
        # x = nn.functional.relu(self.conv5(x))
        # x = nn.functional.max_pool2d(x, 2)
        x = x.view(-1, 64 * 26 * 26)
        x = nn.functional.relu(self.fc1(x))
        # x = nn.functional.relu(self.fc2(x))
        # x = self.fc3(x)
        return x


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data.sampler import SubsetRandomSampler

# 读取数据集，这里假设数据集已经被处理成了符合Pytorch要求的格式
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_dataset = ImageFolder('./train_sample', transform=data_transforms)
test_dataset = ImageFolder('./test_sample', transform=data_transforms)

n_classes = len(train_dataset.classes)

# 划分数据集
val_size = 0.2
shuffle_dataset = True
random_seed = 42
dataset_size = len(train_dataset)
indices = list(range(dataset_size))
split = int(val_size * dataset_size)
if shuffle_dataset:
    torch.manual_seed(random_seed)
    indices = torch.randperm(dataset_size)
train_indices, val_indices = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

# 创建数据加载器
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
val_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=val_sampler)

if __name__ == "__main__":

    # 定义模型和优化器
    net = Net()
    criterion = nn.CrossEntropyLoss()

    learning_rate = 0.01

    optimizer = optim.Adam(net.parameters(), learning_rate)

    # 开始训练
    n_epochs = 10
    best_val_acc = 0
    for epoch in range(n_epochs):
        net.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        net.eval()
        n_correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = net(inputs)
                _, predictions = torch.max(outputs, dim=1)
                n_correct += (predictions == labels).sum().item()
                total += len(labels)
        val_acc = n_correct / total
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(net.state_dict(), 'modelX.pth')
        print(f'Epoch {epoch+1}/{n_epochs}, Val Acc: {val_acc:.4f}, Best Val Acc: {best_val_acc:.4f}')