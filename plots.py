import os
import matplotlib.pyplot as plt
import numpy as np
import time  # Import time for measuring execution duration

# Create output directory if it doesn't exist
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)


def save_plot(fig, filename):
    fig.savefig(os.path.join(output_dir, filename))
    plt.close(fig)


def plot_mean_std(outputs):
    fig, ax = plt.subplots(figsize=(12, 8))

    for name, output in outputs.items():
        mean_output = output.mean(dim=0).detach().cpu().numpy()  # (context_len, embed_dim)
        std_output = output.std(dim=0).detach().cpu().numpy()

        ax.plot(mean_output[:, 0], label=f'Mean {name}')
        ax.fill_between(range(mean_output.shape[0]),
                        mean_output[:, 0] - std_output[:, 0],
                        mean_output[:, 0] + std_output[:, 0],
                        alpha=0.2)

    ax.set_title('Mean and Standard Deviation of MHA Outputs')
    ax.set_xlabel('Token Index')
    ax.set_ylabel('Output Value')
    ax.legend()
    ax.grid()
    save_plot(fig, 'mean_std_plot.png')


def plot_histogram(outputs):
    fig, ax = plt.subplots(figsize=(12, 8))

    for name, output in outputs.items():
        flattened_output = output.detach().cpu().numpy().flatten()
        ax.hist(flattened_output, bins=50, alpha=0.5, label=name)

    ax.set_title('Histogram of MHA Outputs')
    ax.set_xlabel('Output Value')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid()
    save_plot(fig, 'histogram_plot.png')


def plot_boxplot(outputs):
    fig, ax = plt.subplots(figsize=(12, 8))
    data = [output.detach().cpu().numpy().flatten() for output in outputs.values()]

    ax.boxplot(data, labels=outputs.keys())
    ax.set_title('Boxplot of MHA Outputs')
    ax.set_ylabel('Output Value')
    ax.grid()
    save_plot(fig, 'boxplot.png')


def plot_line_chart(outputs):
    fig, ax = plt.subplots(figsize=(12, 8))

    for name, output in outputs.items():
        ax.plot(output.detach().cpu().numpy().mean(axis=0), label=name)

    ax.set_title('Mean Output Line Chart for Each MHA Method')
    ax.set_xlabel('Token Index')
    ax.set_ylabel('Output Value')
    ax.legend()
    ax.grid()
    save_plot(fig, 'line_chart_plot.png')


def plot_heatmap(outputs):
    fig, ax = plt.subplots(figsize=(12, 8))

    for name, output in outputs.items():
        mean_output = output.mean(dim=0).detach().cpu().numpy()  # (context_len, embed_dim)
        cax = ax.imshow(mean_output, aspect='auto', cmap='viridis', alpha=0.5)
        fig.colorbar(cax, ax=ax, label='Output Value')
        ax.set_title(f'Heatmap of Mean Outputs - {name}')
        ax.set_xlabel('Embedding Dimension')
        ax.set_ylabel('Token Index')
        save_plot(fig, f'heatmap_{name}.png')
        ax.cla()  # Clear axis for the next iteration


def plot_speed_comparison(speed_data):
    fig, ax = plt.subplots(figsize=(12, 8))

    methods = list(speed_data.keys())
    times = [speed_data[method] for method in methods]

    bars = ax.bar(methods, times, color='skyblue', edgecolor='black', linewidth=1.5)

    # Add labels on top of the bars
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.2f}',
                ha='center', va='bottom', fontsize=10, color='black')

    # Set titles and labels with a larger font size
    ax.set_title('Speed Comparison of MHA Methods', fontsize=16, fontweight='bold')
    ax.set_ylabel('Time (seconds)', fontsize=14)
    ax.set_xlabel('MHA Method', fontsize=14)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')

    # Enhance gridlines
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_ylim(0, max(times) * 1.1)  # Add some space above the highest bar

    # Add a background color
    fig.patch.set_facecolor('lightgrey')

    # Add a tight layout for better spacing
    plt.tight_layout()

    # Save the plot
    save_plot(fig, 'speed_comparison_plot.png')


import torch
import torch.nn as nn
from models.causal_attention import Ch03_MHA_Wrapper
from models.multi_head_attention import Ch03_MHA  # Existing MHA class
from models.multi_head_attention_combined import MultiHeadAttentionCombinedQKV  # New combined QKV class
from models.mha_einsum import MHAEinsum  # Importing the MHAEinsum class
from models.mha_pytorch_scaled import MHAPyTorchScaledDotProduct  # Importing the MHA PyTorch scaled dot product class
import plots  # Import the custom plots module

# Set random seed for reproducibility
torch.manual_seed(123)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define parameters
batch_size = 8
context_len = 1024
embed_dim = 768

# Generate random embeddings
embeddings = torch.randn((batch_size, context_len, embed_dim), device=device)

# Initialize MHA methods
mha_methods = {
    "MHA Wrapper": Ch03_MHA_Wrapper(
        d_in=embed_dim,
        d_out=embed_dim // 12,
        context_length=context_len,
        dropout=0.0,
        num_heads=12,
        qkv_bias=False
    ).to(device),
    "Ch03_MHA": Ch03_MHA(
        d_in=embed_dim,
        d_out=embed_dim,
        context_length=context_len,
        dropout=0.0,
        num_heads=12,
        qkv_bias=False
    ).to(device),
    "Combined QKV MHA": MultiHeadAttentionCombinedQKV(
        d_in=embed_dim,
        d_out=embed_dim,
        num_heads=12,
        context_length=context_len,
        dropout=0.0,
        qkv_bias=False
    ).to(device),
    "MHA Einsum": MHAEinsum(
        d_in=embed_dim,
        d_out=embed_dim,
        context_length=context_len,
        dropout=0.0,
        num_heads=12,
        qkv_bias=False
    ).to(device),
    "MHA PyTorch Scaled Dot Product": MHAPyTorchScaledDotProduct(
        d_in=embed_dim,
        d_out=embed_dim,
        context_length=context_len,
        dropout=0.0,
        num_heads=12,
        qkv_bias=False
    ).to(device)
}

outputs = {}  # Initialize the outputs dictionary
speed_data = {}  # Initialize the speed data dictionary

# Forward pass through each MHA method and collect outputs and execution time
for name, mha in mha_methods.items():
    start_time = time.time()  # Start timing
    out = mha(embeddings)  # Forward pass
    elapsed_time = time.time() - start_time  # Calculate elapsed time

    outputs[name] = out  # Store the output
    speed_data[name] = elapsed_time  # Store the execution time

# Call the plotting functions
plots.plot_mean_std(outputs)  # Mean and std plot
plots.plot_histogram(outputs)  # Histogram plot
plots.plot_boxplot(outputs)  # Boxplot
plots.plot_line_chart(outputs)  # Line chart
plots.plot_heatmap(outputs)  # Heatmap
plots.plot_speed_comparison(speed_data)  # Speed comparison plot


