import os
import ipyparallel as ipp
import time
import pickle

from neuralhydrology.utils.config import Config

from neuralhydrology.nh_run import ESDL_start_run, eval_run
from pathlib import Path
import xarray as xr
from neuralhydrology.utils.nh_results_ensemble import create_results_ensemble

mycluster = ipp.Cluster(n=int(os.getenv('SLURM_CPUS_ON_NODE')))
c = mycluster.start_and_connect_sync()
dview = c[:]
dview.block = True   

print(f'Number of CPUs: {len(c.ids)}')

hidden_states = range(63, 63+len(c.ids))
dropout_rate = [0.5]
input_seq_lengths = [90]

param_combos = []
for i in hidden_states:
    for j in dropout_rate:
        for k in input_seq_lengths:
            param_combos.append((i, j, k))

print(f'Number of models trained: {len(param_combos)}')

def ESDL_ensemble(params):
    '''Train ensemble for given config file and return ensemble NSE'''
    num_ensemble_members=1
    
    #copy data 24 times, each worker calls a different one
    
    config_path = Path("../initial_exploration/parallel_grid_search.yml") #read this in once, pass in object to all workers
    config = Config(config_path)
    config.update_config({'epochs': 15})
    config.update_config({'hidden_size': params[0]})
    config.update_config({'output_dropout': params[1]})
    config.update_config({'seq_length': params[2]})
    
    output_path = 'parallel_grid_search_test' + str(params[0]) + '_' + str(params[1]) + '_' + str(params[2])
    config.update_config({'experiment_name': output_path})
    
    #train num_ensemble_members models
    paths = [] #store the path of the results of the model
    for i in range(num_ensemble_members):
        ESDL_start_run(config, gpu=-1)
        path = config.run_dir
        paths.append(path)
    
    #evaluate models
    for p in paths:
        eval_run(run_dir=p, period="test")
        eval_run(run_dir=p, period="validation")
        with open(p / "test" / "model_epoch015" / "test_results.p", "rb") as fp: #comment next three lines out when using more than one ensemble member
            results = pickle.load(fp)
            ensemble_nse = results['Tuler']['1D']['NSE'] 

    # ensemble_run = create_results_ensemble(paths, period='validation')
    # ensemble_nse = ensemble_run['Tuler']['1D']['NSE']
    return (params, ensemble_nse)

dview.execute('from neuralhydrology.utils.config import Config')
dview.execute('from neuralhydrology.nh_run import ESDL_start_run, eval_run')
dview.execute('from neuralhydrology.nh_run import ESDL_start_run, eval_run')
dview.execute('from pathlib import Path')
dview.execute('import xarray as xr')
dview.execute('from neuralhydrology.utils.nh_results_ensemble import create_results_ensemble')
dview.execute('import pickle')

lview = c.load_balanced_view()
# Cause execution on main process to wait while tasks sent to workers finish
lview.block = True 

start_time = time.time()

all_nse = lview.map(ESDL_ensemble, param_combos) # map each param combo to the ESDL_ensemble fn, where they run in parallel
end_time = time.time()

elapsed_time = end_time - start_time
print("Elapsed time:", elapsed_time, "seconds")

print(f"Results: {all_nse}")