import torch
import torch.nn as nn

from model import CIFAR10CNN
from dataset import get_dataloaders
from utils import accuracy

def train():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    train_loader,val_loader,test_loader = get_dataloaders(32)

    model = CIFAR10CNN().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=20,
        gamma=0.1
    )

    loss_fn = nn.CrossEntropyLoss()

    epochs = 40

    for epoch in range(epochs):

        model.train()

        running_loss = 0

        for images,labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = loss_fn(outputs,labels)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        scheduler.step()

        val_acc = accuracy(model,val_loader,device)

        print(
            f"Epoch {epoch+1} "
            f"Loss {running_loss/len(train_loader):.4f} "
            f"Val Acc {val_acc:.2f}%"
        )

    test_acc = accuracy(model,test_loader,device)

    print(f"\nFinal Test Accuracy: {test_acc:.2f}%")
    torch.save(model.state_dict(), "cifar10_model.pth")
    print("Model saved as cifar10_model.pth")

if __name__ == "__main__":

    train()
    