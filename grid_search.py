import itertools
import subprocess
import os

from concurrent.futures import ThreadPoolExecutor

# Defining the grid
param_grid = {
    'beta_max': [1.0, 1.5],
    'beta_cycles': [4, 8],
    'learning_rate': [3e-4, 1e-3, 3e-3],
    'n_filters': [3, 4],
    'kernel_v': [5, 7],
    'kernel_h': [5, 7],
    'stride_v': [1, 2],
    'stride_h': [1, 2]
}

# Launching a single training process
def run_experiment(run_id, params):
    run_name = f"run_{run_id:04d}"
    
    # Building the terminal command
    keys = list(param_grid.keys())
    cmd = ["python", TRAIN_SCRIPT]
    for key, value in zip(keys, params):
        cmd.extend([f"--{key}", str(value)])
        
    cmd.extend(["--run_name", run_name])
    
    print(f"[{run_name}] Started")
    
    # Piping output to a text file
    os.makedirs("grid_logs", exist_ok=True)
    with open(f"grid_logs/{run_name}.log", "w") as log_file:
        subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT)
        
    print(f"[{run_name}] Finished")

if __name__ == "__main__":
    # Generating all combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))
    
    # Executing in parallel
    TRAIN_SCRIPT = "train.py"
    MAX_GPU_WORKERS = 16
    print(f"Total combinations to run: {len(combinations)}")
    print(f"Running {MAX_GPU_WORKERS} jobs in parallel\n")
    
    with ThreadPoolExecutor(max_workers=MAX_GPU_WORKERS) as executor:
        for i, params in enumerate(combinations):
            executor.submit(run_experiment, i+1, params)
            
    print("Grid search completed")