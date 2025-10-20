import pandas as pd
import os
import glob
from pathlib import Path

def assemble_results_table(dataset_name):
    """
    Assemble results from individual experiment directories into a table
    similar to Table V showing the effect of lambda and rho (p).
    
    Args:
        dataset_name: 'MNIST' or 'FMNIST'
    """
    base_path = f'/home/iftikhar/Documents/KHU/WINLAB/papers-implementations/StudentAccount/FL-JointPG--inference-stage-optimization-WirelessEdge/code/results/SimpleCNN/{dataset_name}'
    
    # Lambda values to search for
    lambda_values = [0.0, 0.1, 0.2, 0.3, 0.5, 0.9]
    
    # Rho (p) values to search for
    rho_values = [0.0, 0.2, 0.4, 0.6, 0.8]
    
    # ETH values available in the data
    eth_values = [0.05, 0.10, 0.20, 0.40, 0.80, 1.20, 1.60, 2.30]
    
    results = []
    
    # For each lambda value
    for lambda_val in lambda_values:
        # Find the corresponding experiment directory
        pattern = f"splitgp_method_splitgp_rounds_120_clients_50_gamma_0.5_lambda_split_{lambda_val}_ETH_0.05"
        exp_dir = os.path.join(base_path, pattern)
        
        if not os.path.exists(exp_dir):
            print(f"Directory not found: {exp_dir}")
            continue
        
        print(f"Processing lambda={lambda_val} from {exp_dir}")
        
        # For each ETH value, read the corresponding CSV
        for eth_val in eth_values:
            csv_file = os.path.join(exp_dir, f"splitgp_combined_results_eth_{eth_val:.2f}_gamma_0.5_lambda_split_{lambda_val}.csv")
            
            if not os.path.exists(csv_file):
                print(f"  CSV not found: {csv_file}")
                continue
            
            # Read the CSV file
            df = pd.read_csv(csv_file)
            
            # Extract results for each rho value
            for rho_val in rho_values:
                # Find the row with matching p value
                row = df[df['p'] == rho_val]
                
                if not row.empty:
                    selective_acc = row['selective_acc'].values[0]
                    results.append({
                        'lambda': lambda_val,
                        'rho': rho_val,
                        'eth': eth_val,
                        'selective_acc': selective_acc,  # Already in percentage
                        'full_acc': row['full_acc'].values[0],
                        'client_acc': row['client_acc'].values[0]
                    })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Create pivot tables for different perspectives
    
    # Table 1: Lambda vs Rho for a specific ETH value (e.g., ETH=0.8)
    eth_fixed = 0.80
    table1_data = results_df[results_df['eth'] == eth_fixed]
    if not table1_data.empty:
        pivot_table1 = table1_data.pivot_table(
            values='selective_acc', 
            index='lambda', 
            columns='rho',
            aggfunc='first'
        )
        
        # Format the table
        output_file1 = os.path.join(base_path, f'table_lambda_vs_rho_eth{eth_fixed}_{dataset_name}.csv')
        pivot_table1.to_csv(output_file1)
        print(f"\nTable 1 saved to: {output_file1}")
        print(f"\nTable: Effect of λ vs ρ (ETH={eth_fixed}) for {dataset_name}")
        print("="*80)
        print(pivot_table1.to_string(float_format=lambda x: f'{x:.2f}%'))
    
    # Table 2: Lambda vs ETH for a specific rho value (e.g., rho=0.4)
    rho_fixed = 0.4
    table2_data = results_df[results_df['rho'] == rho_fixed]
    if not table2_data.empty:
        pivot_table2 = table2_data.pivot_table(
            values='selective_acc', 
            index='lambda', 
            columns='eth',
            aggfunc='first'
        )
        
        output_file2 = os.path.join(base_path, f'table_lambda_vs_eth_rho{rho_fixed}_{dataset_name}.csv')
        pivot_table2.to_csv(output_file2)
        print(f"\nTable 2 saved to: {output_file2}")
        print(f"\nTable: Effect of λ vs ETH (ρ={rho_fixed}) for {dataset_name}")
        print("="*80)
        print(pivot_table2.to_string(float_format=lambda x: f'{x:.2f}%'))
    
    # Table 3: Full results (all combinations)
    output_file3 = os.path.join(base_path, f'table_all_results_{dataset_name}.csv')
    results_df.to_csv(output_file3, index=False)
    print(f"\nFull results saved to: {output_file3}")
    
    # Create formatted table similar to Table V (Lambda vs Rho for different ETH values)
    print(f"\n\nCreating formatted tables for different ETH values...")
    for eth_val in [0.05, 0.20, 0.40, 0.80, 1.20, 1.60, 2.30]:
        table_data = results_df[results_df['eth'] == eth_val]
        if not table_data.empty:
            pivot_table = table_data.pivot_table(
                values='selective_acc', 
                index='lambda', 
                columns='rho',
                aggfunc='first'
            )
            
            output_file = os.path.join(base_path, f'table_V_format_eth{eth_val}_{dataset_name}.csv')
            pivot_table.to_csv(output_file)
            print(f"  ETH={eth_val}: {output_file}")
            print(pivot_table.to_string(float_format=lambda x: f'{x:.2f}%'))
            print()
    
    return results_df

if __name__ == "__main__":
    print("="*80)
    print("ASSEMBLING RESULTS FOR MNIST")
    print("="*80)
    mnist_results = assemble_results_table('MNIST')
    
    print("\n\n")
    print("="*80)
    print("ASSEMBLING RESULTS FOR FMNIST")
    print("="*80)
    fmnist_results = assemble_results_table('FMNIST')
    
    print("\n\nAll results assembled successfully!")
