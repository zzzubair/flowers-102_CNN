import argparse

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import Flowers102

from model import FlowerClassifier
from train import TEST_TRANSFORM, evaluate_model


def load_model(checkpoint, device):
    model = FlowerClassifier().to(device)
    # The checkpoint is a tensor-only state dictionary. weights_only prevents
    # arbitrary Python objects from being constructed while loading it.
    state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="state_dict/state_dictionary.pth")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = Flowers102(args.data_dir, split="test", download=True, transform=TEST_TRANSFORM)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    evaluate_model(load_model(args.checkpoint, device), loader, device)


if __name__ == "__main__":
    main()
