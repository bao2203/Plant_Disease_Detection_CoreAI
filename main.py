import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.model import PlantCNN
from src.dataset import PlantDataset
from src.train import train_one_epoch
from src.evaluate import evaluate_model


def main():
    # =============== CONFIG ===============
    data_path = "data"          # FOLDER CONTAINS EACH CLASSES FOLDER
    batch_size = 8
    learning_rate = 0.001
    epochs = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # =============== TRANSFORM ===============
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    # =============== LOAD DATASET ===============
    dataset = PlantDataset(root=data_path, transform=transform)

    # 80% train – 20% val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # =============== MODEL ===============
    model = PlantCNN(num_classes=len(dataset.class_names)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    # =============== TRAIN ===============
    print("Start training...\n")
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_acc = evaluate_model(model, val_loader, device)

        print(f"[Epoch {epoch+1}/{epochs}]  Loss: {train_loss:.4f}  |  Val Acc: {val_acc:.2f}%")

    # =============== SAVE MODEL ===============
    torch.save(model.state_dict(), "plant_model.pth")
    print("\nModel saved as plant_model.pth")

    # =============== TEST EXAMPLE IMAGES ===============
    test_path = "test.jpg"   # empty if there's zero images
    try:
        from PIL import Image
        img = Image.open(test_path).convert("RGB")
        img = transform(img).unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
            output = model(img)
            pred = torch.argmax(output, dim=1).item()

        print("\nPrediction for test.jpg:", dataset.class_names[pred])
    except:
        print("\nUnable to find test.jpg, skip prediction.")

    print("\nDone!")


if __name__ == "__main__":
    main()
