import json
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from src.dataset import get_dataloaders
from src.model import get_model

def load_config(config_path: str) -> dict:
    with open(config_path) as file:
        return yaml.safe_load(file)

def create_training_components(config: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(
        architecture=config["model"]["architecture"],
        num_classes=config["model"]["num_classes"],
    ).to(device)

    train_loader, val_loader = get_dataloaders(
        data_dir=config["data"]["data_dir"],
        batch_size=config["training"]["batch_size"],
        num_workers=0,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
    )

    criterion = nn.CrossEntropyLoss()

    return model, train_loader, val_loader, optimizer, criterion, device


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(0)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


def log_metrics(epoch: int, train_loss: float, train_accuracy: float,
                val_loss: float, val_accuracy: float) -> None:
    metrics = {
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
    }

    print(json.dumps(metrics))


def save_checkpoint(
    model: nn.Module,
    checkpoint_dir: str,
    model_name: str,
) -> Path:
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    model_path = checkpoint_path / model_name
    torch.save(model.state_dict(), model_path)

    return model_path


def should_stop_early(
    val_loss: float,
    best_val_loss: float,
    patience_counter: int,
    patience: int,
) -> tuple[bool, float, int]:
    if val_loss < best_val_loss:
        return False, val_loss, 0

    patience_counter += 1

    should_stop = patience_counter >= patience

    return should_stop, best_val_loss, patience_counter



def main(config_path: str = "configs/training_config.yaml") -> None:
    config = load_config(config_path)

    (
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device,
    ) = create_training_components(config)

    best_val_loss = float("inf")
    patience_counter = 0

    training_config = config["training"]
    output_config = config["output"]

    for epoch in range(1, training_config["epochs"] + 1):
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        log_metrics(
            epoch,
            train_loss,
            train_accuracy,
            val_loss,
            val_accuracy,
        )

        if val_loss < best_val_loss:
            save_checkpoint(
                model,
                output_config["checkpoint_dir"],
                output_config["model_name"],
            )

        should_stop, best_val_loss, patience_counter = should_stop_early(
            val_loss,
            best_val_loss,
            patience_counter,
            training_config["early_stopping_patience"],
        )

        if should_stop:
            print(json.dumps({
                "early_stopping": True,
                "epoch": epoch,
            }))
            break


if __name__ == "__main__":
    main()