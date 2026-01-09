#!/usr/bin/env python3
"""
Aggregate and summarize results from all ablation experiments.
"""
import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def load_experiment_results(output_dir):
    """Load results from all completed experiments."""
    output_dir = Path(output_dir)
    logs_dir = output_dir / "logs"
    config_file = Path(__file__).parent / "experiment_configs.json"
    
    with open(config_file, 'r') as f:
        experiments = json.load(f)
    
    results = []
    
    for exp in experiments:
        exp_name = exp['name']
        # Find the log directory for this experiment
        exp_log_dir = logs_dir / exp_name
        
        if not exp_log_dir.exists():
            print(f"Warning: No logs found for {exp_name}")
            continue
        
        # Find the latest version directory
        version_dirs = sorted(exp_log_dir.glob("version_*"), key=lambda x: int(x.name.split("_")[1]))
        if not version_dirs:
            print(f"Warning: No version directories found for {exp_name}")
            continue
        
        log_dir = version_dirs[-1]  # Use most recent version
        metrics_file = log_dir / "metrics.csv"
        
        if not metrics_file.exists():
            print(f"Warning: Metrics file not found for {exp_name}")
            continue
        
        try:
            metrics = pd.read_csv(metrics_file)
            val_metrics = metrics[metrics["val_loss"].notna()]
            
            if len(val_metrics) == 0:
                print(f"Warning: No validation metrics for {exp_name}")
                continue
            
            # Get final validation loss
            final_val_loss = val_metrics["val_loss"].iloc[-1]
            min_val_loss = val_metrics["val_loss"].min()
            
            # Get training metrics
            train_metrics = metrics[metrics["train_loss"].notna()]
            final_train_loss = train_metrics["train_loss"].iloc[-1] if len(train_metrics) > 0 else None
            
            results.append({
                'name': exp_name,
                'model_type': exp['model_type'],
                'use_position': exp['use_position'],
                'use_velocity': exp['use_velocity'],
                'use_acceleration': exp['use_acceleration'],
                'use_intent': exp['use_intent'],
                'final_val_loss': final_val_loss,
                'min_val_loss': min_val_loss,
                'final_train_loss': final_train_loss,
                'num_features': sum([exp['use_position']*2, exp['use_velocity']*2, exp['use_acceleration']*2]) + (1 if exp['use_intent'] else 0),
            })
        except Exception as e:
            print(f"Error processing {exp_name}: {e}")
            continue
    
    return pd.DataFrame(results)

def create_summary_plots(df, output_dir):
    """Create summary plots comparing experiments."""
    output_dir = Path(output_dir)
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Validation loss by model type
    plt.figure(figsize=(12, 6))
    for model_type in df['model_type'].unique():
        subset = df[df['model_type'] == model_type]
        plt.scatter(subset['num_features'], subset['min_val_loss'], 
                   label=model_type, alpha=0.7, s=100)
    plt.xlabel('Number of Features')
    plt.ylabel('Minimum Validation Loss')
    plt.title('Validation Loss vs Number of Features')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(viz_dir / "loss_vs_features.png", dpi=200)
    plt.close()
    
    # Plot 2: Comparison by feature combination
    plt.figure(figsize=(14, 8))
    df_sorted = df.sort_values('min_val_loss')
    plt.barh(range(len(df_sorted)), df_sorted['min_val_loss'])
    plt.yticks(range(len(df_sorted)), df_sorted['name'])
    plt.xlabel('Minimum Validation Loss')
    plt.title('Validation Loss by Experiment')
    plt.tight_layout()
    plt.savefig(viz_dir / "loss_comparison.png", dpi=200)
    plt.close()
    
    # Plot 3: Heatmap of feature combinations
    feature_cols = ['use_position', 'use_velocity', 'use_acceleration', 'use_intent']
    pivot_data = df.pivot_table(
        values='min_val_loss',
        index=['model_type'],
        columns=feature_cols,
        aggfunc='mean'
    )
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot_data, annot=True, fmt='.4f', cmap='viridis_r')
    plt.title('Validation Loss Heatmap by Feature Combination')
    plt.tight_layout()
    plt.savefig(viz_dir / "loss_heatmap.png", dpi=200)
    plt.close()

def main():
    output_dir = Path(__file__).parent / "outputs"
    
    print("Loading experiment results...")
    df = load_experiment_results(output_dir)
    
    if len(df) == 0:
        print("No results found!")
        return
    
    print(f"\nLoaded {len(df)} experiment results")
    
    # Save summary CSV
    summary_file = output_dir / "experiment_summary.csv"
    df.to_csv(summary_file, index=False)
    print(f"Summary saved to {summary_file}")
    
    # Print top 5 best experiments
    print("\nTop 5 experiments (by minimum validation loss):")
    print(df.nsmallest(5, 'min_val_loss')[['name', 'model_type', 'min_val_loss', 'num_features']].to_string(index=False))
    
    # Print statistics by model type
    print("\nStatistics by model type:")
    print(df.groupby('model_type')['min_val_loss'].agg(['mean', 'std', 'min', 'max']))
    
    # Create plots
    print("\nCreating summary plots...")
    create_summary_plots(df, output_dir)
    print("Plots saved to outputs/visualizations/")

if __name__ == '__main__':
    main()


