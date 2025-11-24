import wandb
import yaml

# Load the sweep configuration
with open('/project/LSV1M_training/sweep_config.yaml') as f:
    sweep_config = yaml.safe_load(f)

# Initialize the sweep
sweep_id = wandb.sweep(sweep_config, project='sweep_LSV1M')

# Define the function to execute each run
def sweep_train():
    # Your training script is encapsulated in this function
    # Ensure it initializes wandb.init() and fetches config from wandb.config
    exec(open('/project/LSV1M_training/sweep_training_script.py').read())

# Run the sweep
wandb.agent(sweep_id, function=sweep_train, count=100)
