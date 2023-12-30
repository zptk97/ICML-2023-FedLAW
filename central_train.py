import copy
import ssl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from models_dict import densenet, resnet, cnn
import numpy as np

# ssl._create_default_https_context = ssl._create_unverified_context


BATCH_SIZE = 128 * 10
tra_transformer = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)
val_transformer = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)
train_dataset = datasets.CIFAR10(root="/home/Dataset/cifar/", train=True, download=True, transform=tra_transformer)
test_dataset = datasets.CIFAR10(root="/home/Dataset/cifar/", train=False, download=False, transform=val_transformer)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          num_workers=0,
                          shuffle=True)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=BATCH_SIZE,
                         num_workers=0,
                         shuffle=False)

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = cnn.CNNCifar10().to(DEVICE)

optimizer = torch.optim.SGD(model.parameters(), lr=0.08, momentum=0.9,
                                weight_decay=5e-4)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.03, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
criterion = nn.CrossEntropyLoss()


def train(model, train_loader, optimizer, log_interval):
    model.train()
    for batch_idx, (image, label) in enumerate(train_loader):
        image = image.to(DEVICE)
        label = label.to(DEVICE)
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print(
                f"train Epoch: {Epoch} [{batch_idx * len(image)}/{len(train_loader.dataset)}({100. * batch_idx / len(train_loader):.0f}%)]\tTrain Loss: {loss.item()}")


def evaluate(model, test_loader):
    model.eval()
    count = 0
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for image, label in test_loader:
            image = image.to(DEVICE)
            label = label.to(DEVICE)
            output = model(image)
            test_loss += criterion(output, label).item()
            prediction = output.max(1, keepdim=True)[1]
            correct += prediction.eq(label.view_as(prediction)).sum().item()
            count += 1

    # test_loss /= len(test_loader.dataset)
    test_loss /= count
    test_accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, test_accuracy


EPOCHS = 200
global_dict_recorder = []
for Epoch in range(1, EPOCHS + 1):
    train(model, train_loader, optimizer, log_interval=200)
    test_loss, test_accuracy = evaluate(model, test_loader)
    # optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.992
    print(optimizer.param_groups[0]['lr'])
    scheduler.step()
    print(f"\n[EPOCH: {Epoch}]\tTest Loss: {test_loss:.4f}\tTest Accuracy: {test_accuracy} % \n")
    torch.save(model.state_dict(), './central/central_model_0.08.pt')

    weights = copy.deepcopy(model.state_dict())
    flatted_weights = []
    for name_param in weights:
        if len(flatted_weights) == 0:
            flatted_weights = torch.flatten(weights[name_param])
        else:
            flatted_weights = torch.cat((flatted_weights, torch.flatten(weights[name_param])))
    flatted_weights = flatted_weights.to('cpu')
    acc = test_accuracy / 100.0
    loss = test_loss
    global_dict = {'flat_w': flatted_weights, 'accuracy': torch.tensor(acc), 'loss': torch.tensor(loss)}
    global_dict_recorder.append(global_dict)
    np.save("central/central_dict_0.08.npy", np.array(global_dict_recorder))

