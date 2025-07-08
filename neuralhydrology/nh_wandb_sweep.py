#!/usr/bin/env python3
"""
This script configures a sweep and creates it on WandB. 

Use nh_run.py in sweep mode to add an agent to the sweep. An agent will then execute as long as there are configurations
left to try in the sweep. 
"""
import wandb

# This specifies the hyperparameters to sweep over
sweep_config = {
    'method': 'bayes',  # Can be 'grid', 'random', or 'bayes'
    'name': 'optimise_satimg_approach',  # Give your sweep a meaningful name
    'metric': {
        'name': 'valid/avg_total_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'learning_rate': {
            'value': 0.001  # value doesn't change during the sweep
        },
        'epochs': {
            'value': 50
        },
        'hidden_size': {
            'values': [64, 128, 256]  # list of values that are tested
        },
        'output_dropout': {
            'values': [0.0, 0.1, 0.2]
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
    print(f"To run a new sweep agent use the neuralhydrology sweep mode:")
    print(f"python ../../neuralhydrology/nh_run.py sweep --config-file /path/to/your/base_config.yml --sweep-id {sweep_id}")
    print()
    print("Sweep configurations will be saved in a 'sweep_<sweep_name>' directory next to your base config file.")
    
    return sweep_id

if __name__ == "__main__":
    sweep_id = create_sweep() 