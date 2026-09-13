import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from evaluate import load_model
from model import FlowerClassifier
from train import train_model


class FlowerClassifierTest(unittest.TestCase):
    def test_forward_shape(self):
        model = FlowerClassifier().eval()
        with torch.no_grad():
            output = model(torch.randn(2, 3, 224, 224))
        self.assertEqual(output.shape, (2, 102))

    def test_safe_state_dict_round_trip(self):
        model = FlowerClassifier()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pth"
            torch.save(model.state_dict(), checkpoint)
            loaded = load_model(checkpoint, torch.device("cpu"))
        self.assertEqual(set(model.state_dict()), set(loaded.state_dict()))

    def test_bundled_checkpoint_loads(self):
        model = load_model("state_dict/state_dictionary.pth", torch.device("cpu"))
        self.assertEqual(len(model.state_dict()), 30)

    def test_one_epoch_synthetic_training_smoke(self):
        dataset = TensorDataset(torch.randn(2, 3, 224, 224), torch.tensor([0, 1]))
        loader = DataLoader(dataset, batch_size=2)
        model = FlowerClassifier()
        optimizer = optim.SGD(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1)
        trained = train_model(
            model, nn.CrossEntropyLoss(), optimizer, scheduler,
            loader, loader, epochs=1, device=torch.device("cpu"),
        )
        self.assertIs(trained, model)


if __name__ == "__main__":
    unittest.main()
