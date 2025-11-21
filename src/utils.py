import torch
import yaml
from torchvision import transforms
from PIL import Image

# Load config.yml
def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# calculate accuracy
def accuracy(outputs, labels):
    # outputs: tensor [batch, num_classes]
    # labels: tensor [batch]
    _, preds = torch.max(outputs, 1)
    return torch.sum(preds == labels).item() / len(labels)

# save checkpoints
def save_checkpoint(model, optimizer, epoch, path="checkpoint.pth"):
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
    }, path)
    print(f"[INFO] Checkpoint saved to {path}")

def load_checkpoint(model, optimizer, path="checkpoint.pth"):
    data = torch.load(path, map_location="cpu")

    model.load_state_dict(data["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(data["optimizer_state"])
        print(f"[INFO] Checkpoint loaded from {path}")
        return data["epoch"]

# use inference per image
inference_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224,224))
])

def load_image(path, target_size=(224, 224)):
    img = Image.open(path)
    return inference_transform(img).unsqueeze(0)