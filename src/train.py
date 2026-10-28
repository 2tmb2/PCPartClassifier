from copy import deepcopy
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "laptop_parts"
EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
OUTPUT_PATH = ROOT / "models" / (
    f"laptop_parts_resnet18_epochs{EPOCHS}_batchsize{BATCH_SIZE}_"
    f"learningrate{LEARNING_RATE}.pth"
)
VALIDATION_FRACTION = 0.2
SEED = 42
NUM_WORKERS = 0
WEIGHTS = ResNet18_Weights.DEFAULT
WEIGHT_TRANSFORMS = WEIGHTS.transforms()
IMAGE_SIZE = WEIGHT_TRANSFORMS.crop_size[0]
MEAN = WEIGHT_TRANSFORMS.mean
STD = WEIGHT_TRANSFORMS.std


def make_loaders():
    normalize = transforms.Normalize(MEAN, STD)
    evaluation_transform = transforms.Compose([
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=WEIGHT_TRANSFORMS.interpolation,
        ),
        transforms.ToTensor(),
        normalize,
    ])
    train_data = datasets.ImageFolder(
        DATA_DIR,
        transforms.Compose([
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE),
                interpolation=WEIGHT_TRANSFORMS.interpolation,
            ),
            transforms.ToTensor(),
            normalize,
        ]),
    )
    validation_data = datasets.ImageFolder(
        DATA_DIR,
        evaluation_transform,
    )
    if train_data.class_to_idx != validation_data.class_to_idx:
        raise RuntimeError("Training and validation class mappings differ.")

    indices = range(len(train_data))
    train_indices, validation_indices = train_test_split(
        indices,
        test_size=VALIDATION_FRACTION,
        random_state=SEED,
        stratify=train_data.targets,
    )
    return (
        DataLoader(
            Subset(train_data, train_indices),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
        ), # train loader
        DataLoader(
            Subset(validation_data, validation_indices),
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
        ), # validation loader
        train_data.classes,
    )


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    training = optimizer is not None
    model.eval()
    if training:
        model.fc.train()

    total_loss = total_correct = total = 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad()
            predictions = model(images)
            loss = loss_fn(predictions, labels)
            if training:
                loss.backward()
                optimizer.step()

            count = labels.size(0)
            total_loss += loss.item() * count
            total_correct += (predictions.argmax(1) == labels).sum().item()
            total += count
    return total_loss / total, total_correct / total


def train() -> None:
    torch.manual_seed(SEED)
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    train_loader, validation_loader, class_names = make_loaders()

    model = models.resnet18(weights=WEIGHTS)
    for parameter in model.parameters():
        parameter.requires_grad = False # freeze all layers except the final one
    model.fc = nn.Linear(model.fc.in_features, len(class_names)) # creates new final layer for classification
    model.to(device)

    cost_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=LEARNING_RATE)
    best_accuracy = -1.0
    best_state = None

    print(f"Device: {device} | Classes: {len(class_names)}")
    for epoch in range(1, EPOCHS + 1):
        print(f"Beginning Epoch {epoch}/{EPOCHS}")
        train_loss, train_accuracy = run_epoch(
            model, train_loader, cost_function, device, optimizer
        )
        val_loss, val_accuracy = run_epoch(
            model, validation_loader, cost_function, device # no optimizer means it won't set weights
        )
        print(
            f"Epoch {epoch}/{EPOCHS}: "
            f"train loss {train_loss:.4f}, accuracy {train_accuracy:.2%}; "
            f"validation loss {val_loss:.4f}, accuracy {val_accuracy:.2%}"
        )
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": "resnet18",
            "class_names": class_names,
            "image_size": IMAGE_SIZE,
            "image_mean": MEAN,
            "image_std": STD,
            "validation_accuracy": best_accuracy,
            "best_epoch": best_epoch,
            "model_state_dict": best_state,
        },
        OUTPUT_PATH,
    )
    print(f"Best validation accuracy: {best_accuracy:.2%}; saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    train()
