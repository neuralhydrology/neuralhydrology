#!/usr/bin/env python3
"""
Example of how to use wandb sweeps with neuralhydrology.

This example shows how to:
1. Create a sweep configuration
2. Initialize the sweep with wandb
3. Run sweep agents using the 'sweep' mode

Before running this example:
1. Make sure you have wandb installed and logged in: wandb login
2. Have a base config file ready (e.g., from examples/01-Introduction/)
3. Set up your data directories correctly

Usage:
1. First, run this script to create the sweep: python sweep_example.py
2. Then run agents using: python ../../neuralhydrology/nh_run.py sweep --config-file /path/to/base_config.yml

Note: Each sweep run will save its complete configuration file (base config + sweep parameters) 
in a directory called 'sweep_<sweep_name>' next to your base config file. The files will be named 
using the wandb run name (e.g., 'sweep_my_hyperparameter_search/zealous-sweep-1.yml').
"""

import wandb
import yaml

# Define the sweep configuration
# This specifies the hyperparameters to sweep over
sweep_config = {
    'method': 'grid',  # Can be 'grid', 'random', or 'bayes'
    'name': 'my_hyperparameter_search',  # Give your sweep a meaningful name
    'metric': {
        'name': 'valid/median_nse',
        'goal': 'maximize'
    },
    'parameters': {
        'learning_rate': {
            'values': [0.001, 0.0005, 0.0001]
        },
        'hidden_size': {
            'values': [64, 128, 256]  # actually sensible
        },
        'dropout': {
            'values': [0.0, 0.1, 0.2]  # sweep up to 0.5
        },
        'epochs': {  # maybe remove it, or change in to a higher value
            'value': 10  # Fixed value
        }, 
        'target_noise_std': {
            'values': [0.0, 0.001, 0.005, 0.01, 0.05]
        },
        'batch_size': {
            'values': [64, 128, 256]
        }
    }
}

def create_sweep():
    """Create a wandb sweep and return the sweep ID.
    
    Returns
    -------
    str
        The sweep ID that can be used to run agents.
    """
    # Initialize wandb project
    wandb.login()
    
    # Create the sweep
    sweep_id = wandb.sweep(sweep_config, project="neuralhydrology")
    
    print(f"Created sweep with ID: {sweep_id}")
    print(f"To run sweep agents, use:")
    print(f"wandb agent {sweep_id}")
    print(f"OR use the neuralhydrology sweep mode:")
    print(f"python ../../neuralhydrology/nh_run.py sweep --config-file /path/to/your/base_config.yml")
    print()
    print("Sweep configurations will be saved in a 'sweep_<sweep_name>' directory next to your base config file.")
    
    return sweep_id

if __name__ == "__main__":
    sweep_id = create_sweep() 