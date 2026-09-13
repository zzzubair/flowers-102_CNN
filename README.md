# Flowers 102 CNN

A from-scratch PyTorch convolutional neural network for classifying the 102 categories in the [Oxford 102 Flower Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/). The project trains its own model rather than fine-tuning a pretrained network and includes a saved state dictionary for evaluation.

## Model

Images are augmented during training, resized to 224 × 224, and normalized with ImageNet channel statistics. The classifier contains:

- three convolution blocks with 32, 64, and 128 channels;
- batch normalization, ELU activations, and max pooling after each convolution;
- a 512-unit fully connected layer with dropout;
- a 102-class output layer.

Training uses cross-entropy loss, Adam with weight decay, and two learning-rate schedules. The repository's recorded test accuracy is **60.12%**. This is a project result, not a reproduced benchmark from the current environment.

## Repository layout

```text
.
├── train.py                         # download, train, validate, and test
├── evaluate.py                      # evaluate the included model weights
├── requirements.txt                 # captured Python environment
└── state_dict/state_dictionary.pth  # saved model state dictionary
```

## Setup

Python 3.10 or a compatible version is recommended. Create an isolated environment and install the captured dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements include CUDA 12 packages from the original environment. PyTorch can still select CPU at runtime, but for a smaller CPU-only setup install a matching CPU build of PyTorch and torchvision using the official [PyTorch installation guide](https://pytorch.org/get-started/locally/).

## Evaluate the included model

```bash
python evaluate.py
```

`torchvision.datasets.Flowers102` downloads the train, validation, and test splits into `data/` on first use, so internet access and several hundred megabytes of free space are required. Evaluation then loads `state_dict/state_dictionary.pth` and prints test accuracy. Run the command from the repository root because paths are relative.

## Train from scratch

```bash
python train.py
```

The script automatically uses CUDA when available and otherwise uses CPU. It runs two 1,000-epoch phases and can therefore take a long time; the current script reports the final test score but does not save newly trained weights. The included state dictionary is only read by `evaluate.py`.

## Notes and limitations

- The architecture is defined independently in both scripts; keep them synchronized if it changes.
- Dataset splits and transforms are fixed in source rather than exposed as command-line options.
- No random seed is set, so retraining results will vary.
- The bundled result has not been accompanied by a training log or per-class metrics.
