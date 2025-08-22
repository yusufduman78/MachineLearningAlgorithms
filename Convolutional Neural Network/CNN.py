import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# %% device choice

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% dataset prepare and processing

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=35),
    transforms.RandomCrop(size=32,padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

trainset = torchvision.datasets.CIFAR10(root="../datasets",train=True,transform=transform,download=True)
testset = torchvision.datasets.CIFAR10(root="../datasets",train=False,transform=transform,download=True)

trainloader = torch.utils.data.DataLoader(trainset,batch_size=32,shuffle=True)
testloader = torch.utils.data.DataLoader(testset,batch_size=32,shuffle=False)




# %% visualizing (optional)

def get_sample_image(dataset):
    data_iter = iter(dataset)
    images, labels = next(data_iter)
    return images, labels


def visualize(dataset, iters):
    images, labels = get_sample_image(dataset)
    plt.figure(figsize=(3 * iters, 3))
    for i in range(iters):
        img = images[i]
        img = img / 2 + 0.5
        np_img = img.numpy()
        np_img = np.transpose(np_img, (1, 2, 0))

        plt.subplot(1, iters, i + 1)
        plt.imshow(np_img)
        plt.title(f"Label: {labels[i]}")
        plt.axis("off")
    plt.show()

visualize(trainloader,4)

# %% model building and loss,optimizer

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,padding=1) # in_channels = rgb, out channels = filter numbers, 32x32
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout2d(0.2)
        self.conv3 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.dropout2 = nn.Dropout(0.4)
        self.fc1 = nn.Linear(128*4*4,128)
        self.fc2 = nn.Linear(128,72)
        self.fc3 = nn.Linear(72,10)
    def forward(self,x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.dropout1(self.pool(self.relu(self.bn2(self.conv2(x)))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = torch.flatten(x,1)
        x = self.dropout2(self.relu(self.fc1(x)))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def loss_and_optimizer(model,lr = 0.001):

    lossf = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=lr)

    return lossf,optimizer

# %% train







# %% test



