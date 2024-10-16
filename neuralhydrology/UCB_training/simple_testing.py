import sys
sys.path.append('/Users/evanrobert/Documents/ESDL_Research/russian_river/UCB-USACE-LSTMs/')
print(1)
from UCB_train import UCB_trainer
print(2)
from pathlib import Path

print(3)

path_to_csv = Path('/Users/evanrobert/Documents/ESDL_Research/russian_river/UCB-USACE-LSTMs/neuralhydrology/UCB_training')
print(4)
params = {'learning_rate': 0.001, 'batch_size': 256, 'epochs': 16}
num_ensemble_members = 1
print("initizalizing...")
trainer = UCB_trainer(path_to_csv, params, num_ensemble_members)
print("training...")
trainer.train()
print("results...")
trainer.results()