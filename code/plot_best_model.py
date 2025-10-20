"""
Best Model Performance Visualization Script
Creates a clean plot showing the performance of the best model configuration.
Plots Test Accuracy vs ρ for a specific parameter configuration.
"""

import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
import os

# ============================================================================
# CONFIGURATION - EDIT THIS SECTION
# ============================================================================
FILE_PATH = "results/SimpleCNN/MNIST/splitgp_method_splitgp_rounds_120_clients_50_gamma_0.5_lambda_split_0.2_ETH_0.05/splitgp_combined_results_eth_2.30_gamma_0.5_lambda_split_0.2.csv"
DATASET_NAME = "MNIST"
MODEL_NAME = "SimpleCNN"
FIGSIZE = (10, 6)
YLIM = (65, 100)  # Y-axis limits (min, max)
OUTPUT_FILENAME = "best_model_performance.png"  # Leave as None to auto-generate
# ============================================================================

def extract_params_from_filename(filename):
    """Extract ETH, gamma, and lambda values from filename"""
    params = {}

    # Extract ETH value
    eth_match = re.search(r'eth_([\d.]+)', filename)
    if eth_match:
        try:
            params['eth'] = float(eth_match.group(1))
        except ValueError:
            pass

    # Extract gamma value
    gamma_match = re.search(r'gamma_([\d.]+)', filename)
    if gamma_match:
        try:
            params['gamma'] = float(gamma_match.group(1))
        except ValueError:
            pass

    # Extract lambda value - fixed pattern to work with or without .csv
    lambda_match = re.search(r'lambda_split_([\d.]+)', filename)
    if lambda_match:
        try:
            params['lambda'] = float(lambda_match.group(1))
        except ValueError:
            pass

    return params

def plot_best_model_performance(df, params, output_file, dataset_name='', model_name='', figsize=(10, 6), ylim=(65, 100)):
    """Create a clean plot showing Test Accuracy vs ρ for the best model"""

    if df.empty:
        print("❌ ERROR: No data to plot!")
        return

    # Check if required columns exist
    if 'p' not in df.columns or 'selective_acc' not in df.columns:
        print(f"❌ ERROR: Missing required columns (p, selective_acc)")
        print(f"Available columns: {list(df.columns)}")
        return

    # Create figure with white background
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    ax.set_facecolor('white')

    # Group by p and calculate mean accuracy
    grouped = df.groupby('p')['selective_acc'].agg(['mean', 'std'])

    # Plot line with smooth appearance
    ax.plot(grouped.index, grouped['mean'],
           linewidth=3.0,
           color='#2ca02c',  # Green color for best model
           label=f"λ={params.get('lambda', 'N/A')}, ETH={params.get('eth', 'N/A')}, γ={params.get('gamma', 'N/A')}",
           alpha=0.9,
           marker='o',
           markersize=6,
           markeredgewidth=1.5,
           markeredgecolor='white')

    # Optionally add confidence band if std is available
    if not grouped['std'].isna().all() and (grouped['std'] > 0).any():
        ax.fill_between(grouped.index,
                        grouped['mean'] - grouped['std'],
                        grouped['mean'] + grouped['std'],
                        color='#2ca02c', alpha=0.2)

    # Formatting to match reference plot
    ax.set_xlabel('Relative Portion of Out-of-Distribution Test Samples ρ', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)

    # Add title with parameters
    title = 'Best Model Performance - SplitGP'
    title_parts = []
    if dataset_name:
        title_parts.append(dataset_name)
    if model_name:
        title_parts.append(model_name)

    if title_parts:
        title += f" ({', '.join(title_parts)})"

    ax.set_title(title, fontsize=14, fontweight='bold')

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color='gray')

    # Legend styling
    ax.legend(loc='lower left', fontsize=10, framealpha=1.0,
             edgecolor='black', fancybox=False)

    # Set axis limits
    ax.set_ylim(ylim)
    ax.set_xlim([0, 1.0])

    # Set tick parameters
    ax.tick_params(axis='both', which='major', labelsize=10)

    # Add border
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.0)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✅ Saved visualization to '{output_file}'")

    return fig

def print_performance_summary(df, params):
    """Print a summary of the model performance"""
    print("\n" + "="*90)
    print("BEST MODEL PERFORMANCE SUMMARY")
    print("="*90)

    print(f"\nParameters:")
    print(f"  λ (lambda): {params.get('lambda', 'N/A')}")
    print(f"  ETH: {params.get('eth', 'N/A')}")
    print(f"  γ (gamma): {params.get('gamma', 'N/A')}")

    print(f"\nPerformance Metrics:")

    if 'p' in df.columns and 'selective_acc' in df.columns:
        # Define p values to show
        p_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

        print(f"\n{'ρ Value':<15} | {'Accuracy (%)':<15} | {'Std Dev':<15}")
        print("-" * 50)

        grouped = df.groupby('p')['selective_acc'].agg(['mean', 'std'])

        for p in p_values:
            if p in grouped.index:
                acc = grouped.loc[p, 'mean']
                std = grouped.loc[p, 'std'] if not pd.isna(grouped.loc[p, 'std']) else 0.0
                print(f"{p:<15.2f} | {acc:<15.2f} | {std:<15.2f}")
            else:
                # Find closest p value
                closest_p = min(grouped.index, key=lambda x: abs(x - p))
                if abs(closest_p - p) < 0.05:
                    acc = grouped.loc[closest_p, 'mean']
                    std = grouped.loc[closest_p, 'std'] if not pd.isna(grouped.loc[closest_p, 'std']) else 0.0
                    print(f"{p:<15.2f} | {acc:<15.2f} | {std:<15.2f}")
                else:
                    print(f"{p:<15.2f} | {'N/A':<15} | {'N/A':<15}")

        # Overall statistics
        print("\n" + "-" * 50)
        print(f"{'Overall Statistics:':<15}")
        print(f"  Mean Accuracy: {df['selective_acc'].mean():.2f}%")
        print(f"  Max Accuracy:  {df['selective_acc'].max():.2f}%")
        print(f"  Min Accuracy:  {df['selective_acc'].min():.2f}%")
        print(f"  Std Dev:       {df['selective_acc'].std():.2f}%")

    print("="*90)

def main():
    # ============================================================================
    # CONFIGURATION - EDIT THIS SECTION
    # ============================================================================
    FILE_PATH = "results/VGG11/CIFAR10/splitgp_method_splitgp_rounds_800_clients_50_gamma_0.5_lambda_split_0.2_ETH_0.5/splitgp_combined_results_eth_2.30_gamma_0.5_lambda_split_0.2.csv"
    DATASET_NAME = "CIFAR10"
    MODEL_NAME = "VGG11"
    FIGSIZE = (10, 6)
    YLIM = (70, 90)  # Y-axis limits (min, max)
    OUTPUT_FILENAME = "best_model_performance.png"  # Leave as None to auto-generate
    # ============================================================================

    print("="*80)
    print("BEST MODEL PERFORMANCE VISUALIZATION")
    print("="*80)
    print(f"Input file: {FILE_PATH}")
    print(f"Dataset: {DATASET_NAME if DATASET_NAME else 'Not specified'}")
    print(f"Model: {MODEL_NAME if MODEL_NAME else 'Not specified'}")
    print(f"Figure size: {FIGSIZE[0]}x{FIGSIZE[1]}")
    print(f"Y-axis limits: {YLIM[0]} to {YLIM[1]}")
    print("="*80)

    # Check if file exists
    if not os.path.exists(FILE_PATH):
        print(f"\n❌ ERROR: File not found: {FILE_PATH}")
        print("\nPlease check the FILE_PATH variable at the top of this script.")
        return

    # Extract parameters from filename
    filename = os.path.basename(FILE_PATH)
    params = extract_params_from_filename(filename)

    print(f"\nExtracted parameters from filename:")
    print(f"  λ (lambda): {params.get('lambda', 'Not found')}")
    print(f"  ETH: {params.get('eth', 'Not found')}")
    print(f"  γ (gamma): {params.get('gamma', 'Not found')}")

    # Load the CSV file
    try:
        print(f"\nLoading data from: {FILE_PATH}")
        df = pd.read_csv(FILE_PATH)
        print(f"✓ Loaded {len(df)} rows")
        print(f"  Columns: {list(df.columns)}")

        # If parameters not found in filename, try to get them from CSV data
        if 'lambda' not in params and 'lambda' in df.columns:
            lambda_val = df['lambda'].iloc[0] if not df.empty else None
            if lambda_val is not None:
                params['lambda'] = float(lambda_val)
                print(f"  → Found λ (lambda) in CSV: {params['lambda']}")

        if 'eth' not in params and 'eth' in df.columns:
            eth_val = df['eth'].iloc[0] if not df.empty else None
            if eth_val is not None:
                params['eth'] = float(eth_val)
                print(f"  → Found ETH in CSV: {params['eth']}")

        if 'gamma' not in params and 'gamma' in df.columns:
            gamma_val = df['gamma'].iloc[0] if not df.empty else None
            if gamma_val is not None:
                params['gamma'] = float(gamma_val)
                print(f"  → Found γ (gamma) in CSV: {params['gamma']}")

    except Exception as e:
        print(f"\n❌ ERROR: Failed to read file: {e}")
        return

    # Determine output path
    # Always include parameters in filename
    dataset_part = f"_{DATASET_NAME}" if DATASET_NAME else ""
    model_part = f"_{MODEL_NAME}" if MODEL_NAME else ""
    gamma_part = f"_gamma{params.get('gamma', 'NA')}" if 'gamma' in params else ""
    lambda_part = f"_lambda{params.get('lambda', 'NA')}" if 'lambda' in params else ""
    eth_part = f"_eth{params.get('eth', 'NA')}" if 'eth' in params else ""

    if OUTPUT_FILENAME and OUTPUT_FILENAME != "best_model_performance.png":
        # Use custom output filename if specified and not default
        output_filename = OUTPUT_FILENAME
    else:
        # Auto-generate output filename with parameters
        output_filename = f'best_model{dataset_part}{model_part}{gamma_part}{lambda_part}{eth_part}.png'

    # Save in the parent directory of the dataset folder
    # For example: if file is in results/SimpleCNN/FMNIST/..., save to results/SimpleCNN/
    output_dir = os.path.dirname(FILE_PATH)
    # Go up to parent directory (e.g., from FMNIST folder to SimpleCNN folder)
    parent_dir = os.path.dirname(output_dir)
    output_path = os.path.join(parent_dir, output_filename)

    # Create visualization
    plot_best_model_performance(df, params, output_path, DATASET_NAME, MODEL_NAME, FIGSIZE, YLIM)

    # Print performance summary
    print_performance_summary(df, params)

    print("\n" + "="*80)
    print("Analysis complete!")
    print(f"📊 Visualization saved at: {os.path.abspath(output_path)}")
    print("="*80)

if __name__ == '__main__':
    main()
