# %%
# Beispiel from https://nextjournal.com/gkoehler/pytorch-mnist

import matplotlib.pyplot as plt

# PyTorch has dynamic execution graphs, meaning the computation graph is created on the fly.
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import pandas as pd

from BobNet import BobNet
from LeNet import LeNet
from LeNet import LeNet_MaxPool


# %%
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

    # Now let's take a look at some examples. We'll use the test_loader for this.
    # %%
    examples = enumerate(
        test_loader
    )  # The enumerate() function adds a counter to an iterable and returns it (the enumerate object).
    batch_idx, (example_data, example_targets) = next(
        examples
    )  # Returns next item from iterator.

    # ─── Training The Model ───────────────────────────────────────────────────────

    # class LeNet(nn.Module):     def __init__(self, filter_size, filters_number_1,filters_number_2):
    network = LeNet_MaxPool(Index, 20, 50)
    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

    # #On the x-axis we want to display the number of training examples the network has seen during training.
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]

    def train(epoch):
        network.train()
        for batch_idx, (data, target) in enumerate(train_loader):
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

    test_loss = 0
    correct = 0

    def test(test_loss, correct):
        network.eval()

        with torch.no_grad():
            for data, target in test_loader:
                output = network(data)
                test_loss += F.nll_loss(output, target, size_average=False).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        print(
            "\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                test_loss,
                correct,
                len(test_loader.dataset),
                100.0 * correct / len(test_loader.dataset),
            )
        )

    test(test_loss, correct)
    for epoch in range(1, n_epochs + 1):
        train(epoch)
        test(test_loss, correct)

    print("Train Done")

    filename = (
        r"C:/Users/vmanukyan/Documents/dev/thesis/nets/Training_Data/LeNet_MaxPool.txt"
    )
    text = "\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
        test_loss,
        correct,
        len(test_loader.dataset),
        100.0 * correct / len(test_loader.dataset),
    )
    with open(filename, "a") as fd:
        fd.write(f"\n{text}")

    # Save in file
    import pandas as pd

    Kernel_size = 3
    Loss = test_loss
    Accurracy = 100.0 * correct / len(test_loader.dataset)
    filename = (
        r"C:/Users/vmanukyan/Documents/dev/thesis/nets/Training_Data/LeNet_MaxPool.csv"
    )
    df = pd.read_csv(filename, sep=";")
    df.loc[len(df)] = [Kernel_size, Loss, Accurracy]
    df.to_csv(filename, sep=";", index=False)
    df = pd.read_csv(filename, sep=";")


print("Loop Training Kernel size Done ")


# %%
print(df)
# %%


import matplotlib.pyplot as plt

# plt.scatter(loss,accuracy)
plt.plot(df["Kernel"], df["Accuracy"], "r--")
plt.xlabel("Kernel")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Kernel size")
plt.grid(True)
plt.show()


# %%
# TODO save in Datei Accuracy, Execution Time of Training
# TODO read Datei und Plot
