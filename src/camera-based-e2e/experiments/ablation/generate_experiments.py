#!/usr/bin/env python3
"""
Generate experiment configurations for feature ablation study.
Creates 15 combinations (excluding empty set) × 2 models = 30 experiments.
"""
import json
from pathlib import Path

def generate_experiment_configs():
    """
    Generate all experiment configurations.
    Returns list of dicts with experiment parameters.
    """
    experiments = []
    
    # All 15 combinations (excluding empty set where no features are selected)
    feature_combinations = [
        # Single features
        {'pos': True, 'vel': False, 'acc': False, 'intent': False},
        {'pos': False, 'vel': True, 'acc': False, 'intent': False},
        {'pos': False, 'vel': False, 'acc': True, 'intent': False},
        {'pos': False, 'vel': False, 'acc': False, 'intent': True},
        
        # Two features
        {'pos': True, 'vel': True, 'acc': False, 'intent': False},
        {'pos': True, 'vel': False, 'acc': True, 'intent': False},
        {'pos': True, 'vel': False, 'acc': False, 'intent': True},
        {'pos': False, 'vel': True, 'acc': True, 'intent': False},
        {'pos': False, 'vel': True, 'acc': False, 'intent': True},
        {'pos': False, 'vel': False, 'acc': True, 'intent': True},
        
        # Three features
        {'pos': True, 'vel': True, 'acc': True, 'intent': False},
        {'pos': True, 'vel': True, 'acc': False, 'intent': True},
        {'pos': True, 'vel': False, 'acc': True, 'intent': True},
        {'pos': False, 'vel': True, 'acc': True, 'intent': True},
        
        # All four features
        {'pos': True, 'vel': True, 'acc': True, 'intent': True},
    ]
    
    # Generate experiments for both model types
    for model_type in ['base', 'vision']:
        for combo in feature_combinations:
            # Generate experiment name
            name_parts = [model_type]
            if combo['pos']:
                name_parts.append('pos')
            if combo['vel']:
                name_parts.append('vel')
            if combo['acc']:
                name_parts.append('acc')
            if combo['intent']:
                name_parts.append('intent')
            
            exp_name = '_'.join(name_parts)
            
            experiments.append({
                'name': exp_name,
                'model_type': model_type,
                'use_position': combo['pos'],
                'use_velocity': combo['vel'],
                'use_acceleration': combo['acc'],
                'use_intent': combo['intent'],
            })
    
    return experiments

def main():
    experiments = generate_experiment_configs()
    
    # Save to JSON file
    output_file = Path(__file__).parent / 'experiment_configs.json'
    with open(output_file, 'w') as f:
        json.dump(experiments, f, indent=2)
    
    print(f"Generated {len(experiments)} experiment configurations")
    print(f"Saved to {output_file}")
    
    # Print summary
    print("\nExperiment summary:")
    print(f"  - Base model experiments: {sum(1 for e in experiments if e['model_type'] == 'base')}")
    print(f"  - Vision model experiments: {sum(1 for e in experiments if e['model_type'] == 'vision')}")
    print(f"  - Total: {len(experiments)}")
    
    # Print first few as examples
    print("\nFirst 5 experiments:")
    for exp in experiments[:5]:
        flags = []
        if exp['use_position']:
            flags.append('--use_position')
        if exp['use_velocity']:
            flags.append('--use_velocity')
        if exp['use_acceleration']:
            flags.append('--use_acceleration')
        if exp['use_intent']:
            flags.append('--use_intent')
        print(f"  {exp['name']}: --model_type {exp['model_type']} {' '.join(flags)}")

if __name__ == '__main__':
    main()


