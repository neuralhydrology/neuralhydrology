#!/usr/bin/env python3
"""
This script configures a sweep and creates it on WandB. 

Use nh_run.py in sweep mode to add an agent to the sweep. An agent will then execute as long as there are configurations
left to try in the sweep. 
"""
import wandb

"""
The general structure of the sweep_config contains the mandatory keys `method`, `name`, `metric` and `parameters`. 
For a working example please refer to `examples/07-WandB/WandB.ipynb`.
sweep_config = {
    'method': <search method> {bayes|grid|random}
    'name': <name of the sweep>,
    'metric': {
        'name': <name of the metric to optimise for>,
        'goal': <optimisation direction> {minimise|maximise}'
    },
    'parameters': {
        # The parameter space to search over
    }
}
"""
sweep_config = {}

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