# CUDA_VISIBLE_DEVICES=0 python train.py --dataset_path ./datasets/numerical_dataset_v1_size50000.xlsx --experiment_name test_v2 --num_epochs 50 --batch_size 512

import os
import argparse
import pandas as pd
import time

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from transformers import get_scheduler
from sklearn.model_selection import train_test_split

from model import NumericOperationDataset, NumericalNet, NumericalLoss

def define_argparse():
    parser = argparse.ArgumentParser(description='Train NumericalNet')
    parser.add_argument('--dataset_path', type=str, default="./datasets/numerical_dataset_v1.xlsx", help='Path to the dataset')
    parser.add_argument('--output_path', type=str, default="./outputs", help='Path to save the trained model')
    parser.add_argument('--experiment_name', type=str, required=True, help='Name of the model')

    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for dataloader")
    parser.add_argument("--num_warmup_steps", type=int, default=1000, help="Number of warmup steps for scheduler")
    parser.add_argument("--early_stopping_patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay")
    parser.add_argument("--lr_scheduler", type=str, default="cosine", help="Learning rate scheduler")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    
    args = parser.parse_args()
    
    args.model_path = os.path.join(args.output_path, args.experiment_name)
    os.makedirs(args.model_path, exist_ok=True)
    
    args.visualize_path = os.path.join(args.model_path, "visualizations")
    os.makedirs(args.visualize_path, exist_ok=True)
    
    return args

def plot_loss(train_losses, valid_losses, title="Train and Valid Loss per Epoch"):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, marker='o', linestyle='-', color='b', label='Train Loss')
    plt.plot(epochs, valid_losses, marker='o', linestyle='-', color='r', label='Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{args.visualize_path}/train_valid_loss.png")

def preprocess_data(df):
    new_dataset = []
    target_units = set()
    
    for _, row in df.iterrows():
        value1, unit1, operation, value2, unit2 = row
        target_units.add(unit1)
        target_units.add(unit2)
        
        new_dataset.append([value1, unit1, operation, value2, unit2])
    
    target_units = list(target_units)
    target_units = sorted(target_units, key=lambda x: x[-1]) # sort by the last character
    unit_to_idx = {unit: idx for idx, unit in enumerate(target_units)}
    
    target_operations = ["=", "<", ">"]
    operation_to_idx = {operation: idx for idx, operation in enumerate(target_operations)}
    
    return new_dataset, unit_to_idx, operation_to_idx

def main(args):
    df = pd.read_excel(args.dataset_path, engine='openpyxl')
    df, unit_to_idx, operation_to_idx = preprocess_data(df)
    
    train_dataset, valid_dataset = train_test_split(df, test_size=0.2, random_state=args.seed)
    tr_ds = NumericOperationDataset(train_dataset, unit_to_idx, operation_to_idx)
    val_ds = NumericOperationDataset(valid_dataset, unit_to_idx, operation_to_idx)
    tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    model = NumericalNet(unit_to_idx, operation_to_idx, args.device, num_layers=3).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    num_training_steps = len(tr_dl) * args.num_epochs
    scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=num_training_steps
    )
    criterion = NumericalLoss()
    
    count, best_loss = 0, float('inf')
    train_losses, valid_losses = [], []
    
    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0
        start = time.time()

        for batch in tr_dl:
            batch = [tensor.to(args.device) for tensor in batch]
            value1, unit1, value2, unit2, operation = batch
            
            value1_unit1_embedding = model(value1, unit1)
            value2_unit2_embedding = model(value2, unit2)

            loss = criterion(value1_unit1_embedding, value2_unit2_embedding)
            train_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
        train_loss /= len(tr_dl)
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_dl:
                batch = [tensor.to(args.device) for tensor in batch]
                value1, unit1, value2, unit2, operation = batch

                value1_unit1_embedding = model(value1, unit1)
                value2_unit2_embedding = model(value2, unit2)

                loss = criterion(value1_unit1_embedding, value2_unit2_embedding)
                val_loss += loss.item()
            
            val_loss /= len(val_dl)
            valid_losses.append(val_loss)
        
        elapsed = (time.time() - start)*1000
        print(f"Epoch[{epoch+1}/{args.num_epochs}]|LR: {optimizer.param_groups[0]['lr']:.6f}|Train Loss: {train_loss:.6f}|Val Loss: {val_loss:.6f}|Elapsed time: {elapsed:.2f}ms")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), f"{args.model_path}/best_model.pth")
            count = 0
            best_epoch = epoch
        else:
            count += 1
            if count == args.early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # load the best model and move to cpu
    model.load_state_dict(torch.load(f"{args.model_path}/best_model.pth"))
    model.to("cpu")
    torch.save(model, f"{args.model_path}/best_model.pth")
    print(f"Best model found at epoch {best_epoch}")
    
    plot_loss(train_losses, valid_losses, title="Train and Valid Loss per Epoch")
    
    
if __name__ == "__main__":
    args = define_argparse()
    main(args)