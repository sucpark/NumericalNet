# python visualize.py --model_path ./outputs/test_v1 

import os
import pandas as pd
import argparse
import json
import torch
import umap.umap_ as umap
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps

def define_argparse():
    parser = argparse.ArgumentParser(description='Visualize the trained model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    args = parser.parse_args()
    args.output_path = os.path.join(args.model_path, "visualizations")
    return args

def get_digit_range(start, end, step, device):
    numeric_range = torch.arange(start, end, step, dtype=torch.float32).view(-1, 1).to(device)
    return numeric_range

def get_digit_embeddings(model, device, start=0, end=10, step=0.1):
    numeric_range = get_digit_range(start, end, step, device)
    with torch.no_grad():
        numeric_embeddings = model.get_digit_embedding(numeric_range)

    numeric_embeddings = numeric_embeddings.squeeze(1)
    return numeric_embeddings.cpu().numpy()

def get_unit_embeddings(unit, unit_to_idx, model, device):
    with torch.no_grad():
        unit_idx = torch.tensor(unit_to_idx[unit]).to(device)
        unit_embedding = model.get_unit_embedding(unit_idx).cpu().numpy()
    return unit_embedding

def get_numeric_unit_embedding(digit, unit, model):
    with torch.no_grad():
        joint_embedding = model(digit, unit)
    return joint_embedding

def apply_umap(embeddings, n_components=2):
    n_neighbors = min(15, len(embeddings) - 1)
    umap_model = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors)
    reduced = umap_model.fit_transform(embeddings)
    return reduced

def plot_embeddings(embeddings_2d, labels, label_texts, title, colors, output_path, plot_type='scatter', extra_text=None, figsize=(8, 6)):
    
    plt.figure(figsize=figsize)
    
    if plot_type == 'scatter':
        # Scatter plot with groups
        unique_labels = np.unique(labels)
        for label, color, text in zip(unique_labels, colors, label_texts):
            mask = labels == label
            plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], color=color, label=text)
    elif plot_type == 'text':
        # Scatter plot with text annotations
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], color=colors, marker='o', s=150)
        if extra_text:
            for i in range(len(extra_text)):
                plt.text(embeddings_2d[i, 0] + 0.05, embeddings_2d[i, 1] + 0.05, extra_text[i], fontsize=12)

    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.title(title)
    plt.legend() if plot_type == 'scatter' else None
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_digit_embeddings(ranges, model, device, title="2D Visualization of Numeric Embeddings", cmap='tab20'):
    # Generate embeddings for each range
    numeric_embeddings = [get_digit_embeddings(model, device, start, end, step) for start, end, step in ranges]
    embeddings = np.concatenate(numeric_embeddings)
    embeddings_2d = apply_umap(embeddings)

    # Generate labels for each range
    labels_list = []
    for i, embeddings in enumerate(numeric_embeddings):
        labels_list.append(np.ones(embeddings.shape[0]) * i)
    labels = np.concatenate(labels_list)
    unique_labels = np.unique(labels)
    labels_text = [f'{x[0]} to {x[1]}' for x in ranges]

    # Generate colors for each range
    cmap = colormaps[cmap]
    colors = [cmap(i) for i in range(len(ranges))] 

    # Draw Plots
    plot_embeddings(
        embeddings_2d, labels, labels_text, title, 
        colors, f"{args.output_path}/digit_embedding.png"
    )
    
def plot_unit_embeddings(model, units, colors, title="2D Visualization of Unit Embeddings"):
    
    unit_embeddings = []
    for unit in units:
        unit_embeddings.append(get_unit_embeddings(unit, model.unit_to_idx, model, model.device))
    unit_embeddings_2d = apply_umap(unit_embeddings)

    # Draw Plots
    plot_embeddings(
        unit_embeddings_2d, None, None, title, 
        colors[:len(units)], f"{args.output_path}/unit_embedding.png", 'text', extra_text=units
    )
    
def plot_numerical_embeddings(digit_unit_ranges, model, title="2D Visualization of Numeric w/ Digit Embeddings", cmap='tab20'):
    numeric_digit_embeddings = []
    numeric_digit_labels = []
    with torch.no_grad():
        for idx, (unit, start, end, step) in enumerate(digit_unit_ranges):
            temp_digit_unit_range = get_digit_range(start, end, step, model.device)
            for digit in temp_digit_unit_range:
                temp_embedding = model(digit, unit).unsqueeze(0)
                numeric_digit_embeddings.append(temp_embedding.cpu().numpy())
            numeric_digit_labels.append(np.ones(len(temp_digit_unit_range))*idx)

    embeddings = np.concatenate(numeric_digit_embeddings)
    embeddings_2d = apply_umap(embeddings)
    labels = np.concatenate(numeric_digit_labels)
    labels_text = [f'{x[1]}{x[0]} to {x[2]}{x[0]}' for x in digit_unit_ranges]

    # Generate colors for each range
    cmap = colormaps[cmap]
    colors = [cmap(i) for i in range(len(digit_unit_ranges))] 

    # Draw Plots
    plot_embeddings(
        embeddings_2d, labels, labels_text, title, 
        colors, f"{args.output_path}/numerical_embedding.png", 'scatter'
    )

def main(args):

    # load the trained model
    model = torch.load(os.path.join(args.model_path, "best_model.pth"))
    model.to(args.device)
    print("Model loaded successfully")
    
    # get digit embeddings
    ranges = [
        (0, 10, 0.1),
        (10, 20, 0.1),
        (20, 30, 0.1),
        (30, 40, 0.1),
        (40, 50, 0.1),
        (100, 200, 1),
        (200, 300, 1),
        (300, 400, 1),
        (400, 500, 1),
        (1000, 2000, 10),
        (2000, 3000, 10),
        (3000, 4000, 10),
        (4000, 5000, 10)
    ]
    plot_digit_embeddings(
        ranges, 
        model, 
        args.device, 
        title="2D Visualization of Numeric Embeddings"
    )
    
    # get unit embeddings
    units= ['g', 'mg', 'kg', 'ml', 'l', 'cm', 'km', 'mm', 'm']
    colors = ['purple', 'purple', 'purple', 'orange', 'orange', 'pink', 'pink', 'pink', 'pink']
    plot_unit_embeddings(
        model,
        units, 
        colors,
        title="2D Visualization of Unit Embeddings"
    )
    
    # get numerical embeddings
    digit_unit_ranges = [
        ("kg", 0, 10, 0.1),
        ("kg", 10, 20, 0.1),
        ("ml", 0, 10, 0.1),
        ("ml", 10, 30, 0.2),
        ("m", 2, 20, 0.2),
        ("m", 20, 40, 0.2),
    ]
    plot_numerical_embeddings(
        digit_unit_ranges, 
        model, 
        title="2D Visualization of Numerical Embeddings (Digit + Unit)"
    )
    
if __name__ == "__main__":
    args = define_argparse()
    main(args)