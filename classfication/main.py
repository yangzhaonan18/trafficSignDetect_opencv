# -*- coding=utf-8 -*-

# 用于对图片进行分类
# 时间：2019年8月3日21:40:08

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torchvision import transforms
import matplotlib.pyplot as plt
import torchvision

print(torch.__version__)

data_train = pd.read_csv('fashionmnist\\fashion-mnist_train.csv')
data_test = pd.read_csv('fashionmnist\\fashion-mnist_test.csv')
print("data_train.head(): \n", data_train.head())


class MyDataset(Dataset):
    """
      Build your own dataset
    """
    def __init__(self, data, transform=None):
        self.fashion_mnist = list(data.values)
        self.transform = transform
        label, img = [], []
        for one_line in self.fashion_mnist:
            label.append(one_line[0])
            img.append(one_line[1:])
        self.label = np.asarray(label)
        self.img = np.asarray(img).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, CHANNEL).astype('float32')

    def __getitem__(self, item):
        label, img = self.label[item], self.img[item]
        if self.transform is not None:
            img = self.transform(img)

        return label, img

    def __len__(self):
        return len(self.label)


BATCH_SIZE = 50
LR = 0.005
NUM_CLASS = 10
IMAGE_SIZE = 28
CHANNEL = 1
Train_epoch = 5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CLASS_CLOTHING = {0: 'T-shirt/top',
                  1: 'Trouser',
                  2: 'Pullover',
                  3: 'Dress',
                  4: 'Coat',
                  5: 'Sandal',
                  6: 'Shirt',
                  7: 'Sneaker',
                  8: 'Bag',
                  9: 'Ankle boot'}

My_transform = transforms.Compose([
    transforms.ToTensor(),  # default : range [0, 255] -> [0.0,1.0]
])

Train_data = MyDataset(data_train, transform=My_transform)
Test_data = MyDataset(data_test, transform=My_transform)

Train_dataloader = DataLoader(dataset=Train_data,
                              batch_size=BATCH_SIZE,
                              shuffle=False
                              )
Test_dataloader = DataLoader(dataset=Test_data,
                             batch_size=BATCH_SIZE,
                             shuffle=False
                             )

data_iter = iter(Train_dataloader)  # at this time I set batch_size = 4
label, img = next(data_iter)


def imshow(img, title):
    img = torchvision.utils.make_grid(img) / 255
    img = img.numpy().transpose([1, 2, 0])
    plt.imshow(img)
    if title is not None:
        plt.title(title)


imshow(img, [CLASS_CLOTHING[x] for x in label.numpy().tolist()])
plt.show()


class My_Model(nn.Module):
    def __init__(self, num_of_class):
        super(My_Model, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7 * 7 * 32, num_of_class)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


def train():
    model = My_Model(NUM_CLASS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, Train_epoch + 1):
        for batch_id, (label, image) in enumerate(Train_dataloader):
            label, image = label.to(device), image.to(device)
            output = model(image)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_id % 1000 == 0:
                print('Loss :{:.4f} Epoch[{}/{}]'.format(loss.item(), epoch, Train_epoch))
    return model


def test(model):
    with torch.no_grad():
        correct = 0
        total = 0
        for label, image in Test_dataloader:
            image = image.to(device)
            label = label.to(device)
            outputs = model(image)
            predicted = torch.argmax(outputs, dim=1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
        print('Test Accuracy of the model on the test images: {} %'.format(100 * correct / total))


if __name__ == '__main__':
    model = train()
    test(model)
