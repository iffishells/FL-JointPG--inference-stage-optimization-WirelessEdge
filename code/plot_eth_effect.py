"""
ETH Effect Analysis Script
Creates a clean plot showing Test Accuracy vs ρ for different ETH values.
Similar to lambda effect analysis but for ETH parameter.
"""

import pandas as pd
import matplotlib.pyplot as plt
import glob
import re
import numpy as np
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze ETH effect on model performance')
    parser.add_argument('--base_folder', type=str, required=True,
                       help='Base folder containing results (e.g., results/SimpleCNN/MNIST/)')
    parser.add_argument('--rounds', type=int, default=None,
                       help='Number of training rounds (e.g., 120)')
    parser.add_argument('--clients', type=int, default=None,
                       help='Number of clients (e.g., 50)')
    parser.add_argument('--lambda_split', type=float, default=None,
                       help='Specific lambda value to filter (e.g., 0.2). If None, uses all lambda values')
    parser.add_argument('--gamma', type=float, default=None,
                       help='Specific gamma value to filter (e.g., 0.5). If None, uses all gamma values')
    parser.add_argument('--output', type=str, default='eth_effect_plot.png',
                       help='Output filename for the plot')
    parser.add_argument('--dataset_name', type=str, default='',
                       help='Dataset name for plot title (e.g., MNIST, FMNIST)')
    parser.add_argument('--figsize', type=str, default='10,6',
                       help='Figure size as width,height (e.g., 10,6)')
    return parser.parse_args()

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
    
    # Extract lambda value - must come before .csv extension
    lambda_match = re.search(r'lambda_split_([\d.]+)\.csv', filename)
    if lambda_match:
        try:
            params['lambda'] = float(lambda_match.group(1))
        except ValueError:
            pass
    
    return params

def load_results(base_folder, rounds_filter=None, clients_filter=None, lambda_filter=None, gamma_filter=None):
    """Load all CSV files from the base folder and organize by ETH value"""
    
    # Build the search pattern based on provided filters
    if rounds_filter is not None and clients_filter is not None and lambda_filter is not None:
        # Search in specific folder matching rounds, clients, and lambda
        folder_pattern = f'splitgp_method_splitgp_rounds_{rounds_filter}_clients_{clients_filter}_gamma_*_lambda_split_{lambda_filter}_ETH_*'
        pattern = os.path.join(base_folder, folder_pattern, 'splitgp_combined*.csv')
    elif rounds_filter is not None and clients_filter is not None:
        # Search for folders matching rounds and clients
        folder_pattern = f'splitgp_method_splitgp_rounds_{rounds_filter}_clients_{clients_filter}_gamma_*_lambda_split_*_ETH_*'
        pattern = os.path.join(base_folder, folder_pattern, 'splitgp_combined*.csv')
    elif rounds_filter is not None and lambda_filter is not None:
        # Search for folders matching rounds and lambda
        folder_pattern = f'splitgp_method_splitgp_rounds_{rounds_filter}_clients_*_gamma_*_lambda_split_{lambda_filter}_ETH_*'
        pattern = os.path.join(base_folder, folder_pattern, 'splitgp_combined*.csv')
    elif clients_filter is not None and lambda_filter is not None:
        # Search for folders matching clients and lambda
        folder_pattern = f'splitgp_method_splitgp_rounds_*_clients_{clients_filter}_gamma_*_lambda_split_{lambda_filter}_ETH_*'
        pattern = os.path.join(base_folder, folder_pattern, 'splitgp_combined*.csv')
    elif rounds_filter is not None:
        # Search for folders matching rounds only
        folder_pattern = f'splitgp_method_splitgp_rounds_{rounds_filter}_clients_*_gamma_*_lambda_split_*_ETH_*'
        pattern = os.path.join(base_folder, folder_pattern, 'splitgp_combined*.csv')
    elif clients_filter is not None:
        # Search for folders matching clients only
        folder_pattern = f'splitgp_method_splitgp_rounds_*_clients_{clients_filter}_gamma_*_lambda_split_*_ETH_*'
        pattern = os.path.join(base_folder, folder_pattern, 'splitgp_combined*.csv')
    elif lambda_filter is not None:
        # Search for folders matching lambda only
        folder_pattern = f'splitgp_method_splitgp_rounds_*_clients_*_gamma_*_lambda_split_{lambda_filter}_ETH_*'
        pattern = os.path.join(base_folder, folder_pattern, 'splitgp_combined*.csv')
    else:
        # Search all folders
        pattern = os.path.join(base_folder, 'splitgp_method_splitgp_*/splitgp_combined*.csv')
    
    filepaths = glob.glob(pattern)
    
    print(f"Searching in: {base_folder}")
    print(f"Search pattern: {pattern}")
    print(f"Found {len(filepaths)} CSV files")
    
    eth_data = {}
    
    for filepath in filepaths:
        filename = os.path.basename(filepath)
        
        # Extract parameters from filename
        params = extract_params_from_filename(filename)
        
        if 'eth' not in params:
            print(f"  Skipping {filename} - no ETH value found")
            continue
        
        # Apply filters
        if lambda_filter is not None and params.get('lambda') != lambda_filter:
            continue
        
        if gamma_filter is not None and params.get('gamma') != gamma_filter:
            continue
        
        # Read the CSV file
        try:
            df = pd.read_csv(filepath)
            
            # Check if dataframe is empty
            if df.empty:
                print(f"  Skipping {filename} - file is empty")
                continue
            
            # Check if required columns exist
            if 'p' not in df.columns or 'selective_acc' not in df.columns:
                print(f"  Skipping {filename} - missing required columns (p, selective_acc)")
                print(f"    Available columns: {list(df.columns)}")
                continue
            
            eth_val = params['eth']
            
            if eth_val not in eth_data:
                eth_data[eth_val] = []
            
            eth_data[eth_val].append(df)
            print(f"  ✓ Loaded: ETH={eth_val}, λ={params.get('lambda', 'N/A')}, γ={params.get('gamma', 'N/A')} ({len(df)} rows)")
            
        except Exception as e:
            print(f"  ✗ Error reading {filepath}: {e}")
    
    return eth_data

def plot_eth_effect_simple(eth_data, output_file, dataset_name='', lambda_value=None, figsize=(10, 6)):
    """Create a clean plot showing Test Accuracy vs ρ for different ETH values"""
    
    if not eth_data:
        print("No data to plot!")
        return
    
    # Create figure with white background
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    ax.set_facecolor('white')

    # Define specific colors for ETH values (matching the reference plot)
    eth_vals_sorted = sorted(eth_data.keys())
    # Using distinct colors similar to the reference image
    color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    colors = [color_palette[i % len(color_palette)] for i in range(len(eth_vals_sorted))]

    # Plot accuracy vs p for each ETH value
    for idx, eth_val in enumerate(eth_vals_sorted):
        dfs = eth_data[eth_val]
        
        # Combine all dataframes for this ETH
        all_data = pd.concat(dfs, ignore_index=True)
        
        # Group by p and calculate mean accuracy
        if 'p' in all_data.columns and 'selective_acc' in all_data.columns:
            grouped = all_data.groupby('p')['selective_acc'].agg(['mean', 'std'])
            
            # Plot line with smooth appearance
            ax.plot(grouped.index, grouped['mean'],
                   linewidth=2.5,
                   color=colors[idx],
                   label=f'ETH={eth_val}',
                   alpha=0.9)

    # Formatting to match reference plot
    ax.set_xlabel('Relative Portion of Out-of-Distribution Test Samples ρ', fontsize=12)
    ax.set_ylabel('Test Accuracy', fontsize=12)

    # Add title with lambda value if specified
    title = 'Effect of ETH Threshold in SplitGP'
    if dataset_name and lambda_value is not None:
        title += f' ({dataset_name}, λ={lambda_value})'
    elif dataset_name:
        title += f' ({dataset_name})'
    elif lambda_value is not None:
        title += f' (λ={lambda_value})'
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color='gray')

    # Legend styling
    ax.legend(loc='lower left', fontsize=10, framealpha=1.0,
             edgecolor='black', fancybox=False)

    # Set axis limits to match reference plot
    ax.set_ylim([63, 100])
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

def print_summary_table(eth_data):
    """Print a table showing accuracy at different ρ values for each ETH"""
    print("\n" + "="*90)
    print("EFFECT OF ETH - ACCURACY AT DIFFERENT ρ VALUES")
    print("="*90)
    
    # Define p values to show
    p_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    # Print header
    header = f"{'Methods':<15} ||"
    for p in p_values:
        header += f"  ρ = {p:<5}"
    print(header)
    print("-" * 90)
    
    # Print data for each ETH
    for eth_val in sorted(eth_data.keys()):
        dfs = eth_data[eth_val]
        all_data = pd.concat(dfs, ignore_index=True)
        
        row = f"ETH = {eth_val:<9} ||"
        
        if 'p' in all_data.columns and 'selective_acc' in all_data.columns:
            for p in p_values:
                # Find data points close to this p value
                p_data = all_data[np.isclose(all_data['p'], p, atol=0.05)]
                if not p_data.empty:
                    acc = p_data['selective_acc'].mean()
                    row += f"  {acc:6.2f}%"
                else:
                    row += f"  {'N/A':>7}"
        
        print(row)
    
    print("="*90)

def main():
    args = parse_args()
    
    # Parse figure size
    figsize = tuple(map(float, args.figsize.split(',')))
    
    print("="*80)
    print("ETH EFFECT ANALYSIS")
    print("="*80)
    print(f"Base folder: {args.base_folder}")
    print(f"Rounds filter: {args.rounds if args.rounds is not None else 'None (all values)'}")
    print(f"Clients filter: {args.clients if args.clients is not None else 'None (all values)'}")
    print(f"Lambda filter: {args.lambda_split if args.lambda_split is not None else 'None (all values)'}")
    print(f"Gamma filter: {args.gamma if args.gamma is not None else 'None (all values)'}")
    print(f"Figure size: {figsize[0]}x{figsize[1]}")
    print("="*80)
    
    # Load results
    eth_data = load_results(args.base_folder, args.rounds, args.clients, args.lambda_split, args.gamma)
    
    if not eth_data:
        print("\n❌ ERROR: No data found!")
        print(f"Please check that the folder '{args.base_folder}' exists and contains results.")
        print("\nTip: Try listing available folders with:")
        print(f"  ls -la {args.base_folder}")
        return
    
    print(f"\n✅ Found {len(eth_data)} different ETH values: {sorted(eth_data.keys())}")
    
    # Determine output path
    if args.output == 'eth_effect_plot.png' or '/' not in args.output:
        dataset_part = args.dataset_name if args.dataset_name else 'dataset'
        rounds_part = f'_rounds{args.rounds}' if args.rounds is not None else ''
        clients_part = f'_clients{args.clients}' if args.clients is not None else ''
        lambda_part = f'_lambda{args.lambda_split}' if args.lambda_split is not None else ''
        gamma_part = f'_gamma{args.gamma}' if args.gamma is not None else ''
        
        if args.output == 'eth_effect_plot.png':
            output_filename = f'eth_effect_{dataset_part}{rounds_part}{clients_part}{lambda_part}{gamma_part}.png'
        else:
            output_filename = args.output
        
        output_path = os.path.join(args.base_folder, output_filename)
    else:
        output_path = args.output
    
    # Create visualization
    plot_eth_effect_simple(eth_data, output_path, args.dataset_name, args.lambda_split, figsize)

    # Print summary table
    print_summary_table(eth_data)
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print(f"📊 Visualization saved at: {os.path.abspath(output_path)}")
    print("="*80)

if __name__ == '__main__':
    main()
