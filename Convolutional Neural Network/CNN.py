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






# %% train







# %% test



