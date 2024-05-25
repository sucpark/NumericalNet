# python visualize.py --model_path ./outputs/test_v2

import os
import argparse
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
    numeric_range = [[np.round(x, 4)] for x in np.arange(start, end, step)]
    numeric_range = torch.tensor(numeric_range, dtype=torch.float32).to(device)
    return numeric_range

def get_digit_embeddings(model, start=0, end=10, step=0.1):
    numeric_range = get_digit_range(start, end, step, model.device)
    with torch.no_grad():
        numeric_embeddings = model.get_digit_embedding(numeric_range)

    numeric_embeddings = numeric_embeddings.squeeze(1)
    return numeric_embeddings.cpu().numpy()

def get_target_units(unit_list, unit_to_idx, device):
    target_units = [[unit_to_idx[unit]] for unit in unit_list]
    target_units = torch.tensor(target_units, dtype=torch.long).to(device)
    return target_units

def get_unit_embeddings(unit_list, model, device):
    target_units = get_target_units(unit_list, model.unit_to_idx, device)
    model.eval()
    with torch.no_grad():
        unit_embeddings = model.get_unit_embedding(target_units)
    unit_embeddings = unit_embeddings.squeeze(1)
    return unit_embeddings.cpu()

def get_joint_embeddings(digits, units, model):
    model.eval()
    with torch.no_grad():
        joint_embedding = model.get_joint_embedding(digits, units)
    return joint_embedding.cpu()
    
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

def plot_digit_embeddings(embeddings, labels, labels_text, title="2D Visualization of Numeric Embeddings", cmap='tab20'):

    concat_embeddings = np.concatenate(embeddings)
    embeddings_2d = apply_umap(concat_embeddings)

    concat_labels = np.concatenate(labels)

    # Generate colors for each range
    cmap = colormaps[cmap]
    colors = [cmap(i) for i in range(len(labels_text))]

    # Draw Plots
    plot_embeddings(embeddings_2d, concat_labels, labels_text, title, colors, f"{args.output_path}/digit_embedding.png")


def plot_unit_embeddings(unit_list, model, title="2D Visualization of Unit Embedding"):
    unit_embeddings = get_unit_embeddings(unit_list, model, model.device)
    unit_embeddings_2d = apply_umap(unit_embeddings)

    colors, texts = [], []
    for unit in unit_list:
        texts.append(unit)
        if unit in ["kg", "g", "mg"]:
            colors.append("purple")
        elif unit in ["ml", "l"]:
            colors.append("orange")
        elif unit in ["cm", "km", "mm", "m"]:
            colors.append("pink")
    
    # Draw Plots
    plot_embeddings(
        unit_embeddings_2d, None, None, title, 
        colors, f"{args.output_path}/unit_embedding.png", 'text', texts
    )

def plot_joint_embeddings(embeddings, labels, labels_text, title="2D Visualization of Numeric w/ Digit Embeddings", cmap='tab20', advanced=False):

    concat_embeddings = np.concatenate(embeddings)
    embeddings_2d = apply_umap(concat_embeddings)
    concat_labels = np.concatenate(labels)

    # Generate colors for each range
    cmap = colormaps[cmap]
    colors = [cmap(i) for i in range(len(labels_text))] 

    # Draw Plots
    output_fn = "joint_embedding.png" if not advanced else "advanced_joint_embedding.png"
    plot_embeddings(
        embeddings_2d, concat_labels, labels_text, title, 
        colors, f"{args.output_path}/{output_fn}", 'scatter'
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
    digit_embeddings = [get_digit_embeddings(model, start, end, step) for start, end, step in ranges]
    digit_labels = [np.ones(len(x))*idx for idx, x in enumerate(digit_embeddings)]
    digit_labels_text = [f'{x[0]} to {x[1]}' for x in ranges]
    plot_digit_embeddings(
        digit_embeddings,
        digit_labels,
        digit_labels_text,
        title="2D Visualization of Numeric Embeddings"
    )
    
    # get unit embeddings
    units= ['g', 'mg', 'kg', 'ml', 'l', 'cm', 'km', 'mm', 'm']
    plot_unit_embeddings(
        units,
        model,
        title="2D Visualization of Unit Embeddings"
    )
    
    # get numerical embeddings
    digit_unit_ranges = [
        ("kg", 0, 10, 0.1),
        ("kg", 10, 20, 0.1),
        ("ml", 0, 10, 0.1),
        ("ml", 10, 30, 0.2),
        ("m", 2, 22, 0.2),
        ("m", 22, 42, 0.2),
    ]
    
    joint_embeddings, joint_labels = [], []
    for idx, (unit, start, end, step) in enumerate(digit_unit_ranges):
        temp_digit_range = get_digit_range(start, end, step, model.device)
        
        units = [unit]*len(temp_digit_range)
        temp_target_units = get_target_units(units, model.unit_to_idx, model.device)
        temp_joint_embedding = get_joint_embeddings(temp_digit_range, temp_target_units, model)
    
        joint_embeddings.append(temp_joint_embedding)
        joint_labels.append(np.ones(len(temp_digit_range))*idx)
        
    label_texts = [f'{x[1]}{x[0]} to {x[2]}{x[0]}' for x in digit_unit_ranges]
    
    plot_joint_embeddings(
        joint_embeddings, 
        joint_labels,
        label_texts,
        title="2D Visualization of Joint Embeddings (Digit + Unit)"
    )
    
    # get numerical embeddings
    digit_unit_ranges = [
        ("mg", 100, 200, 1),
        ("g", 0.2, 0.3, 0.001),
        ("ml", 400, 500, 1),
        ("l", 0.5, 0.6, 0.001),
        ("cm", 100, 200, 1),
        ("m", 2, 3, 0.01),
    ]
    joint_embeddings, joint_labels = [], []
    for idx, (unit, start, end, step) in enumerate(digit_unit_ranges):
        temp_digit_range = get_digit_range(start, end, step, model.device)
        
        units = [unit]*len(temp_digit_range)
        temp_target_units = get_target_units(units, model.unit_to_idx, model.device)
        temp_joint_embedding = get_joint_embeddings(temp_digit_range, temp_target_units, model)
    
        joint_embeddings.append(temp_joint_embedding)
        joint_labels.append(np.ones(len(temp_digit_range))*idx)
        
    label_texts = [f'{x[1]}{x[0]} to {x[2]}{x[0]}' for x in digit_unit_ranges]
    
    plot_joint_embeddings(
        joint_embeddings, 
        joint_labels,
        label_texts,
        title="2D Visualization of Joint Embeddings (Digit + Unit)",
        advanced=True
    )
    
if __name__ == "__main__":
    args = define_argparse()
    main(args)