import argparse
import copy

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import Flowers102

from model import FlowerClassifier


TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
TEST_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, epochs, device):
    best_weights = copy.deepcopy(model.state_dict())
    best_accuracy = 0.0
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for phase, loader in (("train", train_loader), ("val", val_loader)):
            model.train(phase == "train")
            running_loss = 0.0
            running_correct = 0
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    predictions = outputs.argmax(1)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_correct += (predictions == labels).sum().item()
            accuracy = 100 * running_correct / len(loader.dataset)
            print(f"{phase} Loss: {running_loss / len(loader.dataset):.4f} Acc: {accuracy:.2f}%")
            if phase == "val" and accuracy > best_accuracy:
                best_accuracy = accuracy
                best_weights = copy.deepcopy(model.state_dict())
        scheduler.step()
    model.load_state_dict(best_weights)
    return model


def evaluate_model(model, loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for inputs, labels in loader:
            predictions = model(inputs.to(device)).argmax(1)
            correct += (predictions == labels.to(device)).sum().item()
    accuracy = 100 * correct / len(loader.dataset)
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy


def data_loaders(data_dir, batch_size):
    datasets = {
        "train": Flowers102(data_dir, split="train", download=True, transform=TRAIN_TRANSFORM),
        "val": Flowers102(data_dir, split="val", download=True, transform=TEST_TRANSFORM),
        "test": Flowers102(data_dir, split="test", download=True, transform=TEST_TRANSFORM),
    }
    return tuple(DataLoader(datasets[name], batch_size=batch_size, shuffle=name == "train") for name in ("train", "val", "test"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs-per-phase", type=int, default=1000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = data_loaders(args.data_dir, args.batch_size)
    model = FlowerClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, total_steps=args.epochs_per_phase)
    model = train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, args.epochs_per_phase, device)
    scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.001, step_size_up=5, mode="triangular2", cycle_momentum=False)
    model = train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, args.epochs_per_phase, device)
    evaluate_model(model, test_loader, device)


if __name__ == "__main__":
    main()
