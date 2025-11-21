import torch

def evaluate_model(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for img, label in loader:
            img, label = img.to(device), label.to(device)

            output = model(img)
            pred = torch.argmax(output, dim=1)

            correct += (pred == label).sum().item()
            total += label.size(0)

    return 100 * correct / total
