import torch


def validate_model(model, dataset_loader, device, criterion):
    loss = 0.0
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in dataset_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss += criterion(outputs, labels)
            del images, labels, outputs
        acc = correct / total
    return acc, loss
