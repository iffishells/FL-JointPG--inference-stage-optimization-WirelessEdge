import pandas as pd
import matplotlib.pyplot as plt
import glob
import re
import numpy as np

# Find all CSV files from experiments with different lambda values
results_pattern = 'results/splitgp_vgg11_results_CIFAR10_method_splitgp_*/*.csv'
filepaths = glob.glob(results_pattern)

# Extract lambda values and organize data
lambda_data = {}

# More robust pattern to extract lambda value
pattern = r'lambda_split_([\d]+\.[\d]+)'

for filepath in filepaths:
    filename = filepath.split('/')[-1]

    # Skip files that don't contain lambda_split
    if 'lambda_split_' not in filename:
        continue

    match = re.search(pattern, filename)
    
    if match:
        try:
            lambda_val = float(match.group(1))
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

print(f"Found data for {len(lambda_data)} different lambda values")
print(f"Lambda values: {sorted(lambda_data.keys())}")

if not lambda_data:
    print("No data found! Check your results directory.")
    exit(1)

# Create visualization
plt.figure(figsize=(12, 8))

# Plot 1: Accuracy vs p for different lambda values
plt.subplot(2, 2, 1)
for lambda_val in sorted(lambda_data.keys()):
    dfs = lambda_data[lambda_val]
    if dfs:
        df = dfs[0]  # Take first if multiple
        if 'selective_acc' in df.columns and 'p' in df.columns:
            plt.plot(df['p'], df['selective_acc']*100, marker='o', 
                    label=f'λ={lambda_val}', linewidth=2)

plt.xlabel('p (OOD Fraction)', fontsize=11)
plt.ylabel('Selective Accuracy (%)', fontsize=11)
plt.title('Effect of Lambda on Selective Accuracy', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend()

# Plot 2: Average accuracy vs lambda
plt.subplot(2, 2, 2)
lambda_vals = []
avg_accs = []

for lambda_val in sorted(lambda_data.keys()):
    dfs = lambda_data[lambda_val]
    if dfs:
        df = dfs[0]
        if 'selective_acc' in df.columns:
            lambda_vals.append(lambda_val)
            avg_accs.append(df['selective_acc'].mean() * 100)

plt.plot(lambda_vals, avg_accs, marker='o', linewidth=2, markersize=8)
plt.xlabel('Lambda (λ)', fontsize=11)
plt.ylabel('Average Selective Accuracy (%)', fontsize=11)
plt.title('Average Accuracy vs Lambda', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)

# Plot 3: Client accuracy vs lambda
plt.subplot(2, 2, 3)
lambda_vals_client = []
client_accs = []

for lambda_val in sorted(lambda_data.keys()):
    dfs = lambda_data[lambda_val]
    if dfs:
        df = dfs[0]
        if 'client_acc' in df.columns:
            lambda_vals_client.append(lambda_val)
            client_accs.append(df['client_acc'].mean() * 100)

if client_accs:
    plt.plot(lambda_vals_client, client_accs, marker='s', linewidth=2, 
            markersize=8, color='orange')
    plt.xlabel('Lambda (λ)', fontsize=11)
    plt.ylabel('Average Client Accuracy (%)', fontsize=11)
    plt.title('Client Accuracy vs Lambda', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)

# Plot 4: Full model accuracy vs lambda
plt.subplot(2, 2, 4)
lambda_vals_full = []
full_accs = []

for lambda_val in sorted(lambda_data.keys()):
    dfs = lambda_data[lambda_val]
    if dfs:
        df = dfs[0]
        if 'full_acc' in df.columns:
            lambda_vals_full.append(lambda_val)
            full_accs.append(df['full_acc'].mean() * 100)

if full_accs:
    plt.plot(lambda_vals_full, full_accs, marker='^', linewidth=2, 
            markersize=8, color='green')
    plt.xlabel('Lambda (λ)', fontsize=11)
    plt.ylabel('Average Full Model Accuracy (%)', fontsize=11)
    plt.title('Full Model Accuracy vs Lambda', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lambda_effect_analysis.png', dpi=300, bbox_inches='tight')
print("Saved visualization to 'lambda_effect_analysis.png'")

# Print summary statistics
print("\n" + "="*60)
print("LAMBDA EFFECT SUMMARY")
print("="*60)
for lambda_val in sorted(lambda_data.keys()):
    dfs = lambda_data[lambda_val]
    if dfs:
        df = dfs[0]
        print(f"\nLambda = {lambda_val}:")
        if 'selective_acc' in df.columns:
            print(f"  Avg Selective Acc: {df['selective_acc'].mean()*100:.2f}%")
        if 'client_acc' in df.columns:
            print(f"  Avg Client Acc: {df['client_acc'].mean()*100:.2f}%")
        if 'full_acc' in df.columns:
            print(f"  Avg Full Acc: {df['full_acc'].mean()*100:.2f}%")

plt.show()
