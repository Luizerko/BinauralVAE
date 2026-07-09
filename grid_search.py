import argparse
import copy
import itertools
import math
import os
import sys

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from data.data_processing import BinauralDataset
from models.VAE import VAE
# from models.CVAE import CVAE

# Defining parameter grids
param_grid_mel = {
    'learning_rate': [1e-3],
    'patience_tol': [0.05],
    'beta_max': [0.8],
    'beta_cycles': [8],
    'w_init': ['torch_default', 'he', 'xavier'],
    'latent_dim_pow': [5, 6],
    'n_filters': [3],
    'kernel_v': [5, 7],
    'kernel_h': [5, 7],
    'stride_v': [1, 2],
    'stride_h': [1],
    'pad': [0]
}

param_grid_stft_4ch = {
    'learning_rate': [1e-3, 1e2],
    'patience_tol': [0.05],
    'beta_max': [0.6, 1.0],
    'beta_cycles': [2, 4],
    'w_init': ['torch_default', 'he', 'xavier'],
    'latent_dim_pow': [5],
    'n_filters': [3],
    'kernel_v': [5, 7],
    'kernel_h': [5, 7],
    'stride_v': [1, 2],
    'stride_h': [1, 2],
    'pad': [0]
}

# Worker function
def train_worker(run_id, keys, params, dataset, train_dataset, val_dataset, base_args):
    # Auxiliar heaviside function for beta computation
    def heaviside(x):
        return 1 if x >= 0 else 0
    
    # Isolate args for this specific process
    args = copy.deepcopy(base_args)
    
    # Injecting the grid hyperparameters into the args namespace
    for key, value in zip(keys, params):
        setattr(args, key, value)
    args.run_name = f"run_{run_id:04d}"
    
    print(f"[{args.run_name}] Started")

    # Redirecting outputs to a text file
    os.makedirs(f"grid_logs/{args.dataset_method}", exist_ok=True)
    sys.stdout = open(f"grid_logs/{args.dataset_method}/{args.run_name}.log", "w+")
    sys.stderr = sys.stdout
    print(f"Hyperparameters: {params}\n")
    
    # Setting up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Creating specific dataloaders for every run
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # Initializing model
    sample_shape = dataset[0].shape
    if args.dataset_method == 'mel' or args.dataset_method == 'stft_4ch':
        model = VAE(image_dimensions=(sample_shape[1], sample_shape[2]), image_channels=sample_shape[0], latent_dim_pow=args.latent_dim_pow, n_filters=args.n_filters, ks_v=args.kernel_v, ks_h=args.kernel_h, s_v=args.stride_v, s_h=args.stride_h, pad=args.pad).to(device)
    model.apply(lambda m: model.init_weights(m, method=args.w_init))

    # Setting up optimizer and early stopping
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    patience_limit = int(args.epochs/10)
    patience_counter = 0
    best_val_loss = float('inf')

    # Cleaning log_dir and setting up logger
    log_path = os.path.join(args.log_dir, args.dataset_method, args.run_name)
    try:
        for file in os.listdir(log_path):
            os.remove(os.path.join(log_path, file))
    except:
        pass
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.dataset_method, args.run_name))

    # Training loop
    for epoch in range(1, args.epochs+1):
        model.train()
        train_rec_loss = 0.0
        train_kl_loss = 0.0
        train_total_loss = 0.0

        # Computing beta coefficient to avoid latent space collapse
        # Linear increasing
        # current_beta = args.beta_max * min(1.0, (epoch+1)/(args.epochs/4))
        
        # Purely cyclic
        # current_beta = args.beta_max * (0.6 - 0.5*math.cos(epoch/(args.epochs/args.beta_cycles) * 2*math.pi))

        # Linear increasing + cyclic
        current_beta = args.beta_max * min(1.0, (epoch+1)/(args.epochs/4)) + heaviside(int((epoch+1) - args.epochs/5)) * -0.3*math.cos(epoch/(args.epochs/args.beta_cycles) * 2*math.pi)

        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f'Epoch {epoch}/{args.epochs} [Train]')
        for batch_idx, batch in progress_bar:
            # Forward pass
            optimizer.zero_grad()
            x = batch.to(device)
            rec, mu, logvar = model(x)

            # Loss computation and backward pass
            rec_loss, kl_loss, total_loss = model.loss(rec, x, mu, logvar, beta=current_beta)
            
            total_loss.backward()
            optimizer.step()

            # Batch logging
            train_rec_loss += rec_loss.item()
            train_kl_loss += kl_loss.item()
            train_total_loss += total_loss.item()

        # Computing average training loss for logging
        train_rec_loss = train_rec_loss/len(train_dataloader)
        train_kl_loss = train_kl_loss/len(train_dataloader)
        train_total_loss = train_total_loss/len(train_dataloader)

        # Computing validation loss for logging and early stopping
        model.eval()
        val_rec_loss = 0.0
        val_kl_loss = 0.0
        val_total_loss = 0.0

        progress_bar = tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc=f'Epoch {epoch}/{args.epochs} [Val]')
        with torch.no_grad():
            for batch_idx, batch in progress_bar:
                # Forward pass
                x = batch.to(device)
                rec, mu, logvar = model(x)
                
                # Loss computation
                rec_loss, kl_loss, total_loss = model.loss(rec, x, mu, logvar, beta=current_beta)
                
                # Batch logging
                val_rec_loss += rec_loss.item()
                val_kl_loss += kl_loss.item()
                val_total_loss += total_loss.item()
                
        # Computing average validation loss for logging
        val_rec_loss = val_rec_loss/len(val_dataloader)
        val_kl_loss = val_kl_loss/len(val_dataloader)
        val_total_loss = val_total_loss/len(val_dataloader)

        # Logging to tensorboard
        writer.add_scalar('Loss/Train_Reconstruction', train_rec_loss, epoch)
        writer.add_scalar('Loss/Train_KL_Divergence', train_kl_loss, epoch)
        writer.add_scalar('Loss/Train_Total', train_total_loss, epoch)

        writer.add_scalar('Loss/Val_Reconstruction', val_rec_loss, epoch)
        writer.add_scalar('Loss/Val_KL_Divergence', val_kl_loss, epoch)
        writer.add_scalar('Loss/Val_Total', val_total_loss, epoch)

        writer.add_scalar('Hyperparameters/Cyclical_Beta', current_beta, epoch)

        # Conditionally logging weights and gradients so not to overflow tensorboard
        if epoch % int(args.epochs/10) == 0:
            for name, param in model.named_parameters():
                writer.add_histogram(f'Weights/{name}', param, epoch)
                if param.grad is not None:
                    writer.add_histogram(f'Gradients/{name}', param.grad, epoch)

        # Early stopping, but no saving. We don't need to save models here, we just want to evaluate them 
        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            patience_counter = 0

        elif val_total_loss >= best_val_loss * (1.0 + args.patience_tol):
            patience_counter += 1
        
        if patience_counter >= patience_limit:
            print(f"Early stopping triggered at epoch {epoch}\n")
            break

    writer.close()
    print(f"[{args.run_name}] Finished successfully")


# Helper function to check if a chunk is done
# def chunk_done(futures):
#     done = True
#     for ft in futures:
#         done = done and ft.done()
#     return done


if __name__ == '__main__':
    # Multiprocessing in 'spawn' mode for safe CUDA operation
    mp.set_start_method('spawn', force=True)

    # Parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default='data/dataset/', help="Path to processed .pt dataset directory")
    parser.add_argument("--dataset_method", type=str, default='mel', help="Method of audio processing used for the dataset creation", choices=['mel', 'stft_4ch', 'stft_complex'])
    parser.add_argument("--save_dir", type=str, default='models/checkpoints/', help="Directory to save model weights")
    parser.add_argument("--log_dir", type=str, default='runs/', help="Directory for logging during model training")
    parser.add_argument("--run_name", type=str, default='run_0001', help="Run name for proper Tensorboard visualization")
    parser.add_argument("--train_size", type=float, default=0.9, help="Train size for train/validation split")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for dataloader")
    parser.add_argument("--num_trains", type=int, default=8, help="Number of training sessions to spawn at a time")

    args = parser.parse_args()

    dataset = BinauralDataset(args.dataset_dir, args.dataset_method)
    train_size = int(args.train_size * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Selecting the active grid
    if args.dataset_method == 'mel':
        keys = list(param_grid_mel.keys())
        values = list(param_grid_mel.values())
    elif args.dataset_method == 'stft_4ch':
        keys = list(param_grid_stft_4ch.keys())
        values = list(param_grid_stft_4ch.values())
    
    combinations = list(itertools.product(*values))
    print(f"Total combinations to run: {len(combinations)}")

    # Creating process chunks
    chunks = [combinations[i:i+args.num_trains] for i in range(0, len(combinations), args.num_trains)]

    # Spawning processes, one chunk at a time
    for idx, chunk in enumerate(chunks):
        futures = {}
        with ProcessPoolExecutor(max_workers=args.num_trains, mp_context=mp) as executor:
            for i, params in enumerate(chunk):
                global_idx = (idx*args.num_trains) + (i+1)
                future = executor.submit(train_worker, global_idx, keys, params, dataset, train_dataset, val_dataset, args)
                futures[future] = global_idx

            # Catching potential errors for debugging
            for future in as_completed(futures):
                run_idx = futures[future]
                try:
                    future.result() 
                except Exception as e:
                    print(f"Error in run {run_idx}:")
                    import traceback
                    traceback.print_exc()

    print("Grid search completed")