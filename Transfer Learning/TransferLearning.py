import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from tqdm import tqdm


#  Device
def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


#  Data preparation
def get_dataloaders(batch_size=64):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # CIFAR-10 mean/std
    ])

    transformtest = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    trainset = torchvision.datasets.CIFAR10(root="../datasets", train=True, transform=transform, download=True)
    testset = torchvision.datasets.CIFAR10(root="../datasets", train=False, transform=transformtest, download=True)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainloader, testloader


#  Model preparation
def get_model(device):
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 10)

    return model.to(device)


#  Training
def train_model(model, trainloader, device, epochs=10, lr=0.001, train_fc_only=True):
    loss_function = nn.CrossEntropyLoss()

    if train_fc_only:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
        optimizer = optim.Adam(model.fc.parameters(), lr=lr)
    else:
        for param in model.parameters():
            param.requires_grad = True
        optimizer = optim.Adam(model.parameters(), lr=lr)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    for epoch in tqdm(range(epochs)):
        model.train()
        running_loss = 0.0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()
        print(f"Epoch: {epoch+1}, loss: {running_loss/len(trainloader):.4f}")

    return model


#  Testing
def test_model(model, testloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predict = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predict == labels).sum().item()
    print(f"Accuracy: {100*correct/total:.2f} %")


#  Main
if __name__ == "__main__":
    device = get_device()
    trainloader, testloader = get_dataloaders()
    model = get_model(device)

    print("\n--- Only Last Layer ---")
    model = train_model(model, trainloader, device, epochs=5, lr=0.001, train_fc_only=True)
    test_model(model, testloader, device)

    print("\n--- All Layer Fine Tuning ---")
    model = train_model(model, trainloader, device, epochs=10, lr=1e-4, train_fc_only=False)
    test_model(model, testloader, device)

    torch.save(model.state_dict(), "cifar_resnet18_transfer.pth")

"""
Using device: cuda
  0%|          | 0/5 [00:00<?, ?it/s]
--- Only Last Layer ---
Epoch: 1, loss: 1.0620
 40%|████      | 2/5 [07:43<11:35, 231.94s/it]Epoch: 2, loss: 0.8304
 60%|██████    | 3/5 [11:41<07:49, 234.78s/it]Epoch: 3, loss: 0.7946
Epoch: 4, loss: 0.7819
100%|██████████| 5/5 [19:38<00:00, 235.72s/it]
Epoch: 5, loss: 0.7658
Accuracy: 78.63 %

--- All Layer Fine Tuning ---
 10%|█         | 1/10 [04:11<37:46, 251.86s/it]Epoch: 1, loss: 0.3418
 20%|██        | 2/10 [08:23<33:34, 251.78s/it]Epoch: 2, loss: 0.1912
Epoch: 3, loss: 0.1432
 40%|████      | 4/10 [16:52<25:23, 253.92s/it]Epoch: 4, loss: 0.1159
 50%|█████     | 5/10 [21:09<21:13, 254.76s/it]Epoch: 5, loss: 0.0973
Epoch: 6, loss: 0.0517
 70%|███████   | 7/10 [29:41<12:46, 255.45s/it]Epoch: 7, loss: 0.0375
 80%|████████  | 8/10 [33:54<08:29, 254.85s/it]Epoch: 8, loss: 0.0299
Epoch: 9, loss: 0.0261
100%|██████████| 10/10 [42:23<00:00, 254.34s/it]
Epoch: 10, loss: 0.0215
Accuracy: 96.41 %
"""