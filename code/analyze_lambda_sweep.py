"""
Lambda Effect Analysis Script
Analyzes the effect of different lambda values on model performance.
Replicates Figure 7 from the paper showing lambda's effect on personalization vs generalization.
"""

import pandas as pd
import matplotlib.pyplot as plt
import glob
import re
import numpy as np
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze lambda effect on model performance')
    parser.add_argument('--base_folder', type=str, required=True,
                       help='Base folder containing results (e.g., results/SimpleCNN/MNIST/)')
    parser.add_argument('--eth', type=float, default=None,
                       help='Specific ETH value to filter (e.g., 0.05). If None, uses all ETH values')
    parser.add_argument('--gamma', type=float, default=None,
                       help='Specific gamma value to filter (e.g., 0.5). If None, uses all gamma values')
    parser.add_argument('--output', type=str, default='lambda_effect_analysis.png',
                       help='Output filename for the plot')
    parser.add_argument('--dataset_name', type=str, default='',
                       help='Dataset name for plot title (e.g., MNIST, FMNIST)')
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

def load_results(base_folder, eth_filter=None, gamma_filter=None):
    """Load all CSV files from the base folder and organize by lambda value"""
    
    # Find all splitgp combined results CSV files
    pattern = os.path.join(base_folder, 'splitgp_method_splitgp_*/splitgp_combined*.csv')
    filepaths = glob.glob(pattern)
    
    print(f"Searching in: {base_folder}")
    print(f"Pattern: {pattern}")
    print(f"Found {len(filepaths)} CSV files")
    
    lambda_data = {}
    
    for filepath in filepaths:
        filename = os.path.basename(filepath)
        
        # Extract parameters from filename
        params = extract_params_from_filename(filename)
        
        if 'lambda' not in params:
            continue
        
        # Apply filters
        if eth_filter is not None and params.get('eth') != eth_filter:
            continue
        
        if gamma_filter is not None and params.get('gamma') != gamma_filter:
            continue
        
        # Read the CSV file
        try:
            df = pd.read_csv(filepath)
            lambda_val = params['lambda']
            
            if lambda_val not in lambda_data:
                lambda_data[lambda_val] = []
            
            lambda_data[lambda_val].append(df)
            print(f"  Loaded: {filename} (λ={lambda_val}, γ={params.get('gamma', 'N/A')}, ETH={params.get('eth', 'N/A')})")
            
        except Exception as e:
            print(f"  Error reading {filepath}: {e}")
    
    return lambda_data

def plot_lambda_effect(lambda_data, output_file, dataset_name=''):
    """Create visualization similar to Figure 7 from the paper"""
    
    if not lambda_data:
        print("No data to plot!")
        return
    
    # Create figure with 2x2 subplots
    fig = plt.figure(figsize=(16, 10))
    
    # Main plot: Test Accuracy vs p for different lambda values (like Figure 7)
    ax_main = plt.subplot(2, 2, (1, 2))  # Span top row
    
    # Generate distinct colors for each lambda value
    lambda_vals_sorted = sorted(lambda_data.keys())
    colors = plt.cm.rainbow(np.linspace(0, 1, len(lambda_vals_sorted)))
    
    # Plot accuracy vs p for each lambda value
    for idx, lambda_val in enumerate(lambda_vals_sorted):
        dfs = lambda_data[lambda_val]
        
        # Combine all dataframes for this lambda
        all_data = pd.concat(dfs, ignore_index=True)
        
        # Group by p and calculate mean accuracy
        if 'p' in all_data.columns and 'selective_acc' in all_data.columns:
            grouped = all_data.groupby('p')['selective_acc'].agg(['mean', 'std'])
            
            ax_main.plot(grouped.index, grouped['mean'],
                        marker='o', linewidth=2.5, markersize=8,
                        color=colors[idx], label=f'λ={lambda_val}')
            
            # Add error bands if we have std
            if not grouped['std'].isna().all():
                ax_main.fill_between(grouped.index, 
                                     (grouped['mean'] - grouped['std']),
                                     (grouped['mean'] + grouped['std']),
                                     alpha=0.2, color=colors[idx])
    
    ax_main.set_xlabel('Relative Portion of Out-of-Distribution Test Samples ρ', fontsize=13)
    ax_main.set_ylabel('Test Accuracy', fontsize=13)
    title = f'Effect of λ in SplitGP'
    if dataset_name:
        title += f' ({dataset_name})'
    ax_main.set_title(title, fontsize=14, fontweight='bold')
    ax_main.grid(True, alpha=0.3, linestyle='--')
    ax_main.legend(loc='best', fontsize=11)
    ax_main.set_ylim([0, 100])
    
    # Bottom left: Average accuracy vs lambda
    ax2 = plt.subplot(2, 2, 3)
    lambda_vals = []
    avg_accs = []
    std_accs = []
    
    for lambda_val in lambda_vals_sorted:
        dfs = lambda_data[lambda_val]
        all_accs = []
        for df in dfs:
            if 'selective_acc' in df.columns:
                all_accs.extend(df['selective_acc'].values)
        
        if all_accs:
            lambda_vals.append(lambda_val)
            avg_accs.append(np.mean(all_accs) * 100)
            std_accs.append(np.std(all_accs) * 100)
    
    ax2.errorbar(lambda_vals, avg_accs, yerr=std_accs, 
                marker='o', linewidth=2.5, markersize=10, capsize=5,
                color='steelblue', markerfacecolor='lightblue', markeredgewidth=2)
    ax2.set_xlabel('Lambda (λ)', fontsize=12)
    ax2.set_ylabel('Average Selective Accuracy (%)', fontsize=12)
    ax2.set_title('Average Performance vs λ', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Bottom right: Performance at different p values (table-like visualization)
    ax3 = plt.subplot(2, 2, 4)
    
    # Create a table showing accuracy at key p values
    p_values = [0.0, 0.2, 0.4, 0.6, 0.8]
    table_data = []
    
    for lambda_val in lambda_vals_sorted:
        dfs = lambda_data[lambda_val]
        all_data = pd.concat(dfs, ignore_index=True)
        
        row = [f'λ={lambda_val}']
        for p in p_values:
            # Find closest p value
            if 'p' in all_data.columns and 'selective_acc' in all_data.columns:
                closest_p = all_data.iloc[(all_data['p'] - p).abs().argsort()[:1]]
                if not closest_p.empty:
                    acc = closest_p['selective_acc'].values[0] * 100
                    row.append(f'{acc:.2f}%')
                else:
                    row.append('N/A')
            else:
                row.append('N/A')
        
        table_data.append(row)
    
    # Create table
    ax3.axis('tight')
    ax3.axis('off')
    
    table = ax3.table(cellText=table_data,
                     colLabels=['λ'] + [f'ρ={p}' for p in p_values],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style the header row
    for i in range(len(p_values) + 1):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax3.set_title('Accuracy at Different ρ Values', fontsize=13, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved visualization to '{output_file}'")
    
    return fig

def print_summary(lambda_data):
    """Print detailed summary statistics"""
    print("\n" + "="*80)
    print("LAMBDA EFFECT SUMMARY")
    print("="*80)
    
    for lambda_val in sorted(lambda_data.keys()):
        dfs = lambda_data[lambda_val]
        all_data = pd.concat(dfs, ignore_index=True)
        
        print(f"\nLambda = {lambda_val}:")
        print(f"  Number of experiments: {len(dfs)}")
        print(f"  Total data points: {len(all_data)}")
        
        if 'selective_acc' in all_data.columns:
            mean_acc = all_data['selective_acc'].mean() * 100
            std_acc = all_data['selective_acc'].std() * 100
            min_acc = all_data['selective_acc'].min() * 100
            max_acc = all_data['selective_acc'].max() * 100
            print(f"  Selective Acc: {mean_acc:.2f}% ± {std_acc:.2f}% (min: {min_acc:.2f}%, max: {max_acc:.2f}%)")
        
        if 'client_acc' in all_data.columns:
            mean_acc = all_data['client_acc'].mean() * 100
            std_acc = all_data['client_acc'].std() * 100
            print(f"  Client Acc: {mean_acc:.2f}% ± {std_acc:.2f}%")
        
        if 'full_acc' in all_data.columns:
            mean_acc = all_data['full_acc'].mean() * 100
            std_acc = all_data['full_acc'].std() * 100
            print(f"  Full Acc: {mean_acc:.2f}% ± {std_acc:.2f}%")
        
        # Show performance at different p values
        if 'p' in all_data.columns and 'selective_acc' in all_data.columns:
            print(f"  Accuracy by ρ:")
            for p_val in [0.0, 0.2, 0.4, 0.6, 0.8]:
                p_data = all_data[np.isclose(all_data['p'], p_val, atol=0.05)]
                if not p_data.empty:
                    acc = p_data['selective_acc'].mean() * 100
                    print(f"    ρ={p_val}: {acc:.2f}%")
    
    print("\n" + "="*80)

def main():
    args = parse_args()
    
    print("="*80)
    print("LAMBDA EFFECT ANALYSIS")
    print("="*80)
    print(f"Base folder: {args.base_folder}")
    print(f"ETH filter: {args.eth if args.eth is not None else 'None (all values)'}")
    print(f"Gamma filter: {args.gamma if args.gamma is not None else 'None (all values)'}")
    print("="*80)
    
    # Load results
    lambda_data = load_results(args.base_folder, args.eth, args.gamma)
    
    if not lambda_data:
        print("\nERROR: No data found!")
        print(f"Please check that the folder '{args.base_folder}' exists and contains results.")
        return
    
    print(f"\nFound {len(lambda_data)} different lambda values: {sorted(lambda_data.keys())}")
    
    # Determine output path
    # If output is just a filename (no path), save in the base_folder
    if args.output == 'lambda_effect_analysis.png' or '/' not in args.output:
        # Create a meaningful default filename
        dataset_part = args.dataset_name if args.dataset_name else 'dataset'
        eth_part = f'_eth{args.eth}' if args.eth is not None else ''
        gamma_part = f'_gamma{args.gamma}' if args.gamma is not None else ''

        if args.output == 'lambda_effect_analysis.png':
            output_filename = f'lambda_effect_{dataset_part}{eth_part}{gamma_part}.png'
        else:
            output_filename = args.output

        # Save in the base_folder
        output_path = os.path.join(args.base_folder, output_filename)
    else:
        # User provided a full path
        output_path = args.output

    print(f"\nOutput will be saved to: {output_path}")

    # Create visualization
    plot_lambda_effect(lambda_data, output_path, args.dataset_name)

    # Print summary
    print_summary(lambda_data)
    
    print("\nAnalysis complete!")
    print(f"📊 Visualization saved at: {os.path.abspath(output_path)}")

if __name__ == '__main__':
    main()
