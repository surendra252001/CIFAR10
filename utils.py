import torch

def accuracy(model,loader,device):

    model.eval()

    total = 0
    correct = 0

    with torch.no_grad():

        for images,labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _,pred = torch.max(outputs,1)

            total += labels.size(0)
            correct += (pred == labels).sum().item()

    return 100 * correct / total