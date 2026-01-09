#!/usr/bin/env python3
"""
Quick analysis script for ablation experiment results.
Reads CSV logs and SLURM outputs to summarize experiment status.
"""

import pandas as pd
import json
from pathlib import Path
import sys
from collections import defaultdict

def analyze_experiment(exp_name, logs_dir, slurm_logs_dir):
    """Analyze a single experiment"""
    results = {
        'name': exp_name,
        'status': 'unknown',
        'final_val_loss': None,
        'epochs_completed': 0,
        'slurm_errors': []
    }
    
    # Check for CSV logs
    exp_log_dir = logs_dir / exp_name
    if exp_log_dir.exists():
        versions = sorted(exp_log_dir.glob('version_*'), key=lambda x: int(x.name.split('_')[1]))
        if versions:
            latest_version = versions[-1]
            metrics_file = latest_version / 'metrics.csv'
            if metrics_file.exists():
                try:
                    df = pd.read_csv(metrics_file)
                    if 'val_loss' in df.columns:
                        val_losses = df['val_loss'].dropna()
                        if len(val_losses) > 0:
                            results['final_val_loss'] = float(val_losses.iloc[-1])
                            results['epochs_completed'] = len(val_losses)
                            results['status'] = 'completed' if results['epochs_completed'] > 0 else 'started'
                except Exception as e:
                    results['status'] = f'error_reading_csv: {e}'
    
    # Check SLURM logs for errors
    slurm_files = list(slurm_logs_dir.glob('slurm-*.out')) + list(slurm_logs_dir.glob('slurm-*.err'))
    for slurm_file in slurm_files:
        try:
            content = slurm_file.read_text()
            if exp_name in content:
                if 'ERROR' in content or 'Traceback' in content:
                    # Extract error lines
                    error_lines = [line for line in content.split('\n') if 'ERROR' in line or 'Error:' in line or 'Traceback' in line]
                    if error_lines:
                        results['slurm_errors'].extend(error_lines[:3])  # First 3 error lines
                        if results['status'] == 'unknown':
                            results['status'] = 'failed'
        except:
            pass
    
    return results

def main():
    base_dir = Path(__file__).parent
    logs_dir = base_dir / 'outputs' / 'logs'
    slurm_logs_dir = base_dir / 'outputs' / 'slurm_logs'
    config_file = base_dir / 'experiment_configs.json'
    
    if not config_file.exists():
        print(f"Error: {config_file} not found")
        sys.exit(1)
    
    # Load experiment configs
    with open(config_file, 'r') as f:
        configs = json.load(f)
    
    print("=" * 80)
    print("ABLATION EXPERIMENT RESULTS SUMMARY")
    print("=" * 80)
    print()
    
    all_results = []
    for config in configs:
        exp_name = config['name']
        results = analyze_experiment(exp_name, logs_dir, slurm_logs_dir)
        all_results.append(results)
    
    # Print summary
    print(f"{'Experiment':<40} {'Status':<15} {'Epochs':<8} {'Final Val Loss':<15}")
    print("-" * 80)
    
    for r in all_results:
        val_loss_str = f"{r['final_val_loss']:.4f}" if r['final_val_loss'] is not None else "N/A"
        print(f"{r['name']:<40} {r['status']:<15} {r['epochs_completed']:<8} {val_loss_str:<15}")
    
    # Summary statistics
    print()
    print("=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    completed = sum(1 for r in all_results if r['status'] == 'completed')
    failed = sum(1 for r in all_results if r['status'] == 'failed')
    unknown = sum(1 for r in all_results if r['status'] == 'unknown')
    
    print(f"Total experiments: {len(all_results)}")
    print(f"  Completed: {completed}")
    print(f"  Failed: {failed}")
    print(f"  Unknown/In Progress: {unknown}")
    
    # Show failed experiments
    if failed > 0:
        print()
        print("FAILED EXPERIMENTS:")
        for r in all_results:
            if r['status'] == 'failed' and r['slurm_errors']:
                print(f"  {r['name']}:")
                for err in r['slurm_errors'][:2]:
                    print(f"    {err}")

if __name__ == '__main__':
    main()








