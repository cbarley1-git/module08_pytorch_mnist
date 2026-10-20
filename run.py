import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from utilities import *

def main():
    args = myargs()
    print(f"Training with -> Epochs: {args.num_epochs}, Batch Size: {args.batch_size}, LR: {args.learning_rate}")

    # Download & Load Data
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize Model, Loss, and Optimizer
    model = ConvNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Train
    model, loss_list, acc_train, acc_test = training(
        model, args.num_epochs, train_loader, test_loader, criterion, optimizer
    )

    # Plot
    plot(loss_list, acc_train, acc_test)
    print("Training complete. Plot saved as figure.png")

if __name__ == '__main__':
    main()