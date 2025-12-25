#!/usr/bin/env python3
"""
Training script for U-Net Segmentation
Includes training loop, testing, and model wrapper

Usage:
    # Train with default settings
    uv run python run.py --train

    # Train with custom hyperparameters
    uv run python run.py --train --epochs 50 --lr 1e-4 --weight_decay 0.01

    # Test model (loads checkpoints/best.pt)
    uv run python run.py

    # Custom save directory
    uv run python run.py --train --save_dir my_checkpoints
"""

import os
import json
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
import tomllib
from pathlib import Path

from dataset import get_dataloaders
from model import UNet

CONFIG_PATH = Path(__file__).parent / "config.toml"
with open(CONFIG_PATH, "rb") as f:
    cfg = tomllib.load(f)

def train(
    model,
    train_loader,
    val_loader,
    optimizer="adamw",
    weight_decay=0.1,
    learning_rate=1e-3,
    epochs=100,
    save_path="checkpoints",
    device=None,
):
    # Auto-detect device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
    )

    # Create save directory
    os.makedirs(save_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    writer = SummaryWriter(log_dir=os.path.join(save_path, "runs", timestamp))
    
    print(f"Training on device: {device}")
    print(f"Optimizer: {optimizer}, LR: {learning_rate}, WD: {weight_decay}")
    print("-" * 60)

    best_val_loss = float("inf")
    best_val_acc = 0.0

    # Training loop with progress bar
    epoch_pbar = tqdm(range(epochs), desc="Training", unit="epoch")

    for epoch in epoch_pbar:
        # === TRAIN ===
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        num_batches = 0

        train_pbar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}", leave=False, unit="batch"
        )
        for x, y in train_pbar:
            x, y = x.to(device), (y.squeeze(1).long()-1).to(device)

            opt.zero_grad()
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            loss.backward()
            opt.step()

            acc = (torch.argmax(y_hat, dim=1) == y).float().mean().item()

            train_loss += loss.item()
            train_acc += acc
            num_batches += 1

            train_pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{acc:.4f}"})

        train_loss /= num_batches
        train_acc /= num_batches

        # === VALIDATE ===
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        num_batches = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), (y.squeeze(1).long()-1).to(device)

                y_hat = model(x)
                loss = loss_fn(y_hat, y)
                acc = (torch.argmax(y_hat, dim=1) == y).float().mean().item()

                val_loss += loss.item()
                val_acc += acc
                num_batches += 1

        val_loss /= num_batches
        val_acc /= num_batches

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)

        current_lr = opt.param_groups[0]["lr"]

        writer.add_scalar("Learning_rate", current_lr, epoch)

        # Update progress bar
        epoch_pbar.set_postfix(
            {
                "train_loss": f"{train_loss:.4f}",
                "val_loss": f"{val_loss:.4f}",
                "val_acc": f"{val_acc:.4f}",
                "lr": f"{current_lr:.6f}",
            }
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_path = os.path.join(save_path, "best.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                },
                best_path,
            )
            tqdm.write(
                f"  → Saved best model (val_loss: {best_val_loss:.4f}, val_acc: {best_val_acc:.4f})"
            )

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            last_path = os.path.join(save_path, "last.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "epoch": epoch,
                },
                last_path,
            )

    # Save final checkpoint
    final_path = os.path.join(save_path, "last.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "epoch": epochs - 1,
        },
        final_path,
    )

    writer.close()

    print(f"\n{'-' * 60}")
    print(f"Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    return best_val_loss


def test(model, test_loader, checkpoint_path=None, device=None):
    # Auto-detect device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    model = model.to(device)

    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Checkpoint loaded successfully")

    print(f"Testing on device: {device}")
    print("-" * 60)

    # Test
    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    num_batches = 0

    loss_fn = nn.CrossEntropyLoss()
    test_pbar = tqdm(test_loader, desc="Testing", unit="batch")

    with torch.no_grad():
        for x, y in test_pbar:
            x, y = x.to(device), (y.squeeze(1).long()-1).to(device)

            y_hat = model(x)
            loss = loss_fn(y_hat, y)

            acc = (torch.argmax(y_hat, dim=1) == y).float().mean().item()

            test_loss += loss.item()
            test_acc += acc
            num_batches += 1

            test_pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{acc:.4f}",
                }
            )

    test_loss /= num_batches
    test_acc /= num_batches

    print(f"\n{'-' * 60}")
    print(f"Test Results:")
    print(f"  Loss:        {test_loss:.4f}")
    print(f"  Accuracy:    {test_acc:.4f} ({test_acc*100:.2f}%)")
    print("-" * 60)

    results = {
        "test_loss": test_loss,
        "test_acc": test_acc,
    }

    # Save results to JSON
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"checkpoints/test_results_{timestamp}.json"
    os.makedirs("checkpoints", exist_ok=True)
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {filename}")

    return results

def parse_args():
    parser = argparse.ArgumentParser(description="Train U-Net on Oxford-IIIT Pet")
    
    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    
    # Model
    parser.add_argument("--num_workers", type=int, default=4)
    
    # Misc
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument('--train', action='store_true', help='Enable training mode')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
        
    in_channels = cfg["in_channels"]
    out_channels = cfg["out_channels"]
    num_workers = cfg["num_workers"]

    weight_decay = args.weight_decay
    learning_rate = args.lr
    epochs = args.epochs
    train_mode = args.train
    batch_size = args.batch_size

    model = UNet(
        in_channels=in_channels,
        out_channels=out_channels
    )
    train_loader = get_dataloaders(split="train", batch_size=batch_size, num_workers=num_workers)
    val_loader = get_dataloaders(split="val", batch_size=batch_size,    num_workers=num_workers)

    if train_mode:
        train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer="adamw",
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            epochs=epochs,
            save_path=args.save_dir
        )
    else:
        test_loader = get_dataloaders(split="test", batch_size=batch_size, num_workers=num_workers)

        test(model=model, test_loader=test_loader, checkpoint_path="checkpoints/best.pt")