# %%
# Beispiel from https://nextjournal.com/gkoehler/pytorch-mnist

import time

import matplotlib.pyplot as plt
import pandas as pd
#%%
# PyTorch has dynamic execution graphs, meaning the computation graph is created on the fly.
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from LeNet import LeNet_MaxPool

for Index in range(2, 9):
    # Here the number of epochs defines how many times we'll loop over the complete trai ning dataset,
    # while learning_rate and momentum are hyperparameters for the optimizer we'll be using later on.
    n_epochs = 3
    batch_size_train = 64
    batch_size_test = 1000
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10

    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    # We'll use a batch_size of 64 for training and size 1000 for testing on this dataset.
    # The values 0.1307 and 0.3081 used for the Normalize() transformation below are the global mean and standard deviation of the MNIST dataset

    # ─── Load Datasets ────────────────────────────────────────────────────────────

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            "/files/",
            train=True,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        ),
        batch_size=batch_size_train,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            "/files/",
            train=False,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        ),
        batch_size=batch_size_test,
        shuffle=True,
    )

    start = time.time()

    examples = enumerate(
        test_loader
    )  # The enumerate() function adds a counter to an iterable and returns it (the enumerate object).
    batch_idx, (example_data, example_targets) = next(
        examples
    )  # Returns next item from iterator.

    # ─── Training The Model ───────────────────────────────────────────────────────

    # class LeNet(nn.Module):     def __init__(self, filter_size, filters_number_1,filters_number_2):
    network = LeNet_MaxPool(Index, 10, 10)
    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

    # #On the x-axis we want to display the number of training examples the network has seen during training.
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]

    global_train_batch_counter = [0]

    new_test_accuracy = []
    new_test_losses = []
    new_test_batch_counter = []

    def train(epoch):
        network.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            global_train_batch_counter[0] += 1
            # alle 10 batches wird test() aufgerufen. die liste new_test_accuracy enthält am Ende
            # alle gemessenen accuracies
            if global_train_batch_counter[0] % 10 == 0:
                test_loss, accuracy = test()
                new_test_accuracy.append(accuracy)
                new_test_losses.append(test_loss)
                new_test_batch_counter.append(global_train_batch_counter[0])

            optimizer.zero_grad()
            output = network(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )
                train_losses.append(loss.item())
                train_counter.append(
                    (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset))
                )
                # torch.save(network.state_dict(), r'C:\dev\result_cnn_mnist\model.pth')
                #
                # torch.save(optimizer.state_dict(), r'C:\dev\result_cnn_mnist\optimizer.pth')

    #test_loss = 0
    #correct = 0

    def test():
        network.eval()
        test_loss = 0
        accuracy = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = network(data)
                test_loss += F.nll_loss(output, target, size_average=False).item()
                pred = output.data.max(1, keepdim=True)[1]
                accuracy += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        print(
            "\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                test_loss,
                accuracy,
                len(test_loader.dataset),
                100.0 * accuracy / len(test_loader.dataset),
            )
        )
        return test_loss, accuracy

    test_loss_correct = [0, 0]  # delete this
    for epoch in range(1, n_epochs + 1):  # 3 epochs -> 3* test
        train(epoch)
        # test(test_loss_correct)

    print("Train Done ", Index)
    end = time.time()
    TrainingTime = round(end - start)
    print(TrainingTime)

    filename_txt = (
        r"C:/Users/vmanukyan/Documents/dev/thesis/nets/Training_Data/LeNet_Kernel_Size.txt"
    )
    text = "\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
        test_loss_correct[0],
        test_loss_correct[1],
        len(test_loader.dataset),
        100.0 * test_loss_correct[1] / len(test_loader.dataset),
    )
    Text_Kernel_Size = 'Kernel Size = '+str(Index) +'  .   Training Time = '+ str(TrainingTime) + ' seconds'
    with open(filename_txt, "a") as fd:
        fd.write(f"\n{Text_Kernel_Size}")
        fd.write(f"\n{text}")
        fd.close()

    # Save in file
    import pandas as pd

    Kernel_Size = Index
    Loss = round(test_loss_correct[0],5)
    Accurracy = int(test_loss_correct[1])/100
    filename = (
        r"C:/Users/vmanukyan/Documents/dev/thesis/nets/Training_Data/LeNet_Kernel_Size.csv"
    )
    df = pd.read_csv(filename, sep=";")
    df.loc[len(df)] = [Kernel_Size, Loss, Accurracy, TrainingTime]
    df.to_csv(filename, sep=";", index=False)
    df = pd.read_csv(filename, sep=";")

print(new_test_accuracy)
print(new_test_loss)
print(new_test_batch_counter)

print("Loop Training Kernel Size Done ")


# %%
