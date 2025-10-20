import pandas as pd
import matplotlib.pyplot as plt
import glob
import re
import numpy as np

# Find all CSV files from experiments in the new directory structure
# Look in both SimpleCNN and VGG11 results for MNIST and FMNIST
results_patterns = [
    'results/SimpleCNN/MNIST/splitgp_method_splitgp_*/splitgp_combined*.csv',
    'results/SimpleCNN/FMNIST/splitgp_method_splitgp_*/splitgp_combined*.csv',
    'results/splitgp_vgg11_results_CIFAR10_method_splitgp_*/splitgp_combined*.csv',
    'results/splitgp_vgg11_results_MNIST_method_splitgp_*/splitgp_combined*.csv',
    'results/splitgp_vgg11_results_FMNIST_method_splitgp_*/splitgp_combined*.csv'
]

filepaths = []
for pattern in results_patterns:
    filepaths.extend(glob.glob(pattern))

print(f"Found {len(filepaths)} CSV files")

# Extract lambda values and organize data
lambda_data = {}

for filepath in filepaths:
    filename = filepath.split('/')[-1]
    
    # Extract lambda from the CSV filename (more reliable)
    # Pattern: lambda_split_X.X.csv
    match = re.search(r'lambda_split_([\d.]+)\.csv', filename)
    
    if match:
        lambda_str = match.group(1)
        try:
            lambda_val = float(lambda_str)
        except ValueError:
            print(f"Could not parse lambda from: {filename}")
            continue
        
        # Read the CSV file
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            continue
        
        # Store data by lambda value
        if lambda_val not in lambda_data:
            lambda_data[lambda_val] = []
        
        lambda_data[lambda_val].append(df)
        print(f"Added {filepath} with lambda={lambda_val}")

print(f"\nFound data for {len(lambda_data)} different lambda values")
print(f"Lambda values: {sorted(lambda_data.keys())}")

if not lambda_data:
    print("No data found! Check your results directory.")
    exit(1)

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Accuracy vs p for different lambda values
ax1 = axes[0, 0]
colors = plt.cm.viridis(np.linspace(0, 1, len(lambda_data)))
for idx, lambda_val in enumerate(sorted(lambda_data.keys())):
    dfs = lambda_data[lambda_val]
    if dfs:
        # Average across all dataframes for this lambda
        all_data = pd.concat(dfs)
        grouped = all_data.groupby('p')['selective_acc'].mean()
        ax1.plot(grouped.index, grouped.values*100, marker='o', 
                label=f'λ={lambda_val}', linewidth=2, color=colors[idx])

ax1.set_xlabel('p (OOD Fraction)', fontsize=12)
ax1.set_ylabel('Selective Accuracy (%)', fontsize=12)
ax1.set_title('Effect of Lambda on Selective Accuracy', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='best')

# Plot 2: Average accuracy vs lambda
ax2 = axes[0, 1]
lambda_vals = []
avg_accs = []
std_accs = []

for lambda_val in sorted(lambda_data.keys()):
    dfs = lambda_data[lambda_val]
    if dfs:
        all_accs = []
        for df in dfs:
            if 'selective_acc' in df.columns:
                all_accs.extend(df['selective_acc'].values)
        
        if all_accs:
            lambda_vals.append(lambda_val)
            avg_accs.append(np.mean(all_accs) * 100)
            std_accs.append(np.std(all_accs) * 100)

if avg_accs:
    ax2.errorbar(lambda_vals, avg_accs, yerr=std_accs, marker='o', 
                linewidth=2, markersize=8, capsize=5)
    ax2.set_xlabel('Lambda (λ)', fontsize=12)
    ax2.set_ylabel('Average Selective Accuracy (%)', fontsize=12)
    ax2.set_title('Average Accuracy vs Lambda', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)

# Plot 3: Client accuracy vs lambda
ax3 = axes[1, 0]
lambda_vals_client = []
client_accs = []
std_client = []

for lambda_val in sorted(lambda_data.keys()):
    dfs = lambda_data[lambda_val]
    if dfs:
        all_accs = []
        for df in dfs:
            if 'client_acc' in df.columns:
                all_accs.extend(df['client_acc'].values)
        
        if all_accs:
            lambda_vals_client.append(lambda_val)
            client_accs.append(np.mean(all_accs) )
            std_client.append(np.std(all_accs) )

if client_accs:
    ax3.errorbar(lambda_vals_client, client_accs, yerr=std_client, 
                marker='s', linewidth=2, markersize=8, color='orange', capsize=5)
    ax3.set_xlabel('Lambda (λ)', fontsize=12)
    ax3.set_ylabel('Average Client Accuracy (%)', fontsize=12)
    ax3.set_title('Client Accuracy vs Lambda', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)

# Plot 4: Full model accuracy vs lambda
ax4 = axes[1, 1]
lambda_vals_full = []
full_accs = []
std_full = []

for lambda_val in sorted(lambda_data.keys()):
    dfs = lambda_data[lambda_val]
    if dfs:
        all_accs = []
        for df in dfs:
            if 'full_acc' in df.columns:
                all_accs.extend(df['full_acc'].values)
        
        if all_accs:
            lambda_vals_full.append(lambda_val)
            full_accs.append(np.mean(all_accs) * 100)
            std_full.append(np.std(all_accs) * 100)

if full_accs:
    ax4.errorbar(lambda_vals_full, full_accs, yerr=std_full, 
                marker='^', linewidth=2, markersize=8, color='green', capsize=5)
    ax4.set_xlabel('Lambda (λ)', fontsize=12)
    ax4.set_ylabel('Average Full Model Accuracy (%)', fontsize=12)
    ax4.set_title('Full Model Accuracy vs Lambda', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lambda_effect_analysis.png', dpi=300, bbox_inches='tight')
print("\nSaved visualization to 'lambda_effect_analysis.png'")

# Print summary statistics
print("\n" + "="*70)
print("LAMBDA EFFECT SUMMARY")
print("="*70)
for lambda_val in sorted(lambda_data.keys()):
    dfs = lambda_data[lambda_val]
    if dfs:
        print(f"\nLambda = {lambda_val}:")
        print(f"  Number of experiments: {len(dfs)}")
        
        # Combine all dataframes
        all_data = pd.concat(dfs)
        
        if 'selective_acc' in all_data.columns:
            print(f"  Selective Acc: {all_data['selective_acc'].mean()*100:.2f}% ± {all_data['selective_acc'].std()*100:.2f}%")
        if 'client_acc' in all_data.columns:
            print(f"  Client Acc: {all_data['client_acc'].mean()*100:.2f}% ± {all_data['client_acc'].std()*100:.2f}%")
        if 'full_acc' in all_data.columns:
            print(f"  Full Acc: {all_data['full_acc'].mean()*100:.2f}% ± {all_data['full_acc'].std()*100:.2f}%")

print("\n" + "="*70)
plt.show()
