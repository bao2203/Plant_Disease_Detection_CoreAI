import torch

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for img, label in loader:
        img, label = img.to(device), label.to(device)

        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)
