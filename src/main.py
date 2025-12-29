import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_dataloaders
from model import get_resnet18
from train import train_model
from evaluate import evaluate_model
from model import get_mobilenet_v2


DATA_DIR = "data/raw"
BATCH_SIZE = 32
EPOCHS = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, val_loader, test_loader, classes = get_dataloaders(
    DATA_DIR,
    BATCH_SIZE
)

model = get_resnet18(len(classes))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

model = train_model(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    EPOCHS
)

print("\nResnet Validation Results:")
evaluate_model(model, val_loader, classes, device)

print("\nResnet Test Results:")
evaluate_model(model, test_loader, classes, device)


import os

os.makedirs("results", exist_ok=True)
torch.save(model.state_dict(), "results/fingerprint_classifier.pth")
print("Model saved to results/fingerprint_classifier.pth")

mobilenet = get_mobilenet_v2(len(classes))
mobilenet = mobilenet.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mobilenet.parameters(), lr=1e-4)

mobilenet = train_model(
    mobilenet,
    train_loader,
    criterion,
    optimizer,
    device,
    EPOCHS
)

print("\nMobileNet Validation Results:")
evaluate_model(mobilenet, val_loader, classes, device)

print("\nMobileNet Test Results:")
evaluate_model(mobilenet, test_loader, classes, device)

torch.save(
    mobilenet.state_dict(),
    "results/mobilenet_v2.pth"
)
print("MobileNet model saved to results/mobilenet_v2.pth")