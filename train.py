import os
import argparse
import math
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from data.data_processing import BinauralDataset
from models.VAE import VAE
from models.CVAE import CVAE


# Auxiliar heaviside function for beta computation
def heaviside(x):
    return 1 if x >= 0 else 0


# Main training loop
def train(args):
    # Setting up device and dataset/dataloader
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = BinauralDataset(args.dataset_dir, args.dataset_method)
    
    train_size = int(args.train_size * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # Initializing model
    sample_shape = dataset[0].shape
    if args.dataset_method == 'mel' or args.dataset_method == 'stft_4ch':
        model = VAE(image_dimensions=(sample_shape[1], sample_shape[2]), image_channels=sample_shape[0], latent_dim_pow=args.latent_dim_pow, n_filters=args.n_filters, ks_v=args.kernel_v, ks_h=args.kernel_h, s_v=args.stride_v, s_h=args.stride_h, pad=args.pad).to(device)
    elif args.dataset_method == 'stft_complex':
        model = CVAE(image_dimensions=(sample_shape[1], sample_shape[2]), image_channels=sample_shape[0], latent_dim_pow=args.latent_dim_pow, n_filters=args.n_filters, ks_v=args.kernel_v, ks_h=args.kernel_h, s_v=args.stride_v, s_h=args.stride_h, pad=args.pad).to(device)
    model.apply(lambda m: model.init_weights(m, method=args.w_init))

    # Setting up optimizer, early stopping and logger
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    patience_limit = int(args.epochs / 10)
    patience_counter = 0
    best_val_loss = float('inf')

    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.dataset_method, args.run_name))

    # Training loop
    for epoch in range(1, args.epochs+1):
        model.train()
        train_rec_loss = 0.0
        train_kl_loss = 0.0
        train_total_loss = 0.0

        # Computing beta coefficient to avoid latent space collapse
        # Linear increasing
        # current_beta = args.beta_max * min(1.0, (epoch+1)/(args.epochs/2))
        
        # Purely cyclic
        # current_beta = args.beta_max * (0.6 - 0.5*math.cos(epoch/(args.epochs/args.beta_cycles) * 2*math.pi))
        
        # Linear increasing + cyclic
        current_beta = args.beta_max * min(1.0, (epoch+1)/(args.epochs/3)) + heaviside(int((epoch+1) - args.epochs/3)) * -0.3*math.cos(epoch/(args.epochs/args.beta_cycles) * 2*math.pi)

        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f'Epoch {epoch}/{args.epochs} [Train]')
        for batch_idx, batch in progress_bar:
            # Forward pass, loss computation and backward pass
            optimizer.zero_grad()
            x = batch.to(device)
            
            if args.dataset_method == 'mel' or args.dataset_method == 'stft_4ch':
                rec, mu, logvar = model(x)
                rec_loss, kl_loss, total_loss = model.loss(rec, x, mu, logvar, beta=current_beta)
            elif args.dataset_method == 'stft_complex':
                rec, mu, sigma, delta= model(x)
                rec_loss, kl_loss, total_loss = model.loss(rec, x, mu, sigma, delta, beta=current_beta)
            
            total_loss.backward()
            optimizer.step()

            # Batch logging (without beta coefficient for proper early stopping)
            train_rec_loss += rec_loss.item()
            train_kl_loss += kl_loss.item()
            train_total_loss += rec_loss.item() + kl_loss.item()

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
                # Forward pass and loss computation
                x = batch.to(device)
                
                if args.dataset_method == 'mel' or args.dataset_method == 'stft_4ch':
                    rec, mu, logvar = model(x)
                    rec_loss, kl_loss, total_loss = model.loss(rec, x, mu, logvar, beta=current_beta)
                elif args.dataset_method == 'stft_complex':
                    rec, mu, sigma, delta= model(x)
                    rec_loss, kl_loss, total_loss = model.loss(rec, x, mu, sigma, delta, beta=current_beta)
                
                # Batch logging
                val_rec_loss += rec_loss.item()
                val_kl_loss += kl_loss.item()
                val_total_loss += rec_loss.item() + kl_loss.item()
                
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

        for name, param in model.named_parameters():
            writer.add_histogram(f'Weights/{name}', param, epoch)
            if param.grad is not None:
                writer.add_histogram(f'Gradients/{name}', param.grad, epoch)

        # Early stopping and model saving
        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            patience_counter = 0

            checkpoint_path = os.path.join(args.save_dir, args.dataset_method, args.run_name, 'model_save.pt')
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_total_loss,
            }, checkpoint_path)

            print(f"Saving best model at epoch {epoch}\n")
            
        # If value is patience_tol or more higher than best value, count for patience
        elif val_total_loss >= best_val_loss * (1.0 + args.patience_tol):
            patience_counter += 1
        
        if patience_counter >= patience_limit:
            print(f"Early stopping triggered at epoch {epoch}\n")
            break

    writer.close()


if __name__ == '__main__':
    # Parsing arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_dir", type=str, default='data/dataset/', help="Path to processed .pt dataset directory")
    parser.add_argument("--dataset_method", type=str, default='mel', help="Method of audio processing used for the dataset creation", choices=['mel', 'stft_4ch', 'stft_complex'])
    parser.add_argument("--save_dir", type=str, default='models/checkpoints/', help="Directory to save model weights")
    parser.add_argument("--log_dir", type=str, default='runs/', help="Directory for logging during model training")
    parser.add_argument("--run_name", type=str, default='test_run_0001', help="Run name for proper Tensorboard visualization")

    parser.add_argument("--train_size", type=float, default=0.9, help="Train size for train/validation split")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for optimizer")
    parser.add_argument("--patience_tol", type=float, default=0.02, help="Tolerance value for early stopping")
    parser.add_argument("--beta_max", type=float, default=0.8, help="Maximum beta weight for KL Divergence loss")
    parser.add_argument("--beta_cycles", type=int, default=6, help="Number of cycles beta weight goes through during training")
    parser.add_argument("--w_init", type=str, default='torch_default', help="Weight initialization method", choices=['he', 'xavier', 'torch_default'])
    
    parser.add_argument("--latent_dim_pow", type=int, default=5, help="Size of the latent space (in powers of 2)")
    parser.add_argument("--n_filters", type=int, default=3, help="Number of conv layers")
    parser.add_argument("--kernel_v", type=int, default=7, help="Kernel size vertically")
    parser.add_argument("--kernel_h", type=int, default=5, help="Kernel size horizontally")
    parser.add_argument("--stride_v", type=int, default=2, help="Vertical stride")
    parser.add_argument("--stride_h", type=int, default=1, help="Horizontal stride")
    parser.add_argument("--pad", type=int, default=0, help="Amount of padding")

    parser.add_argument("--num_workers", type=int, default=24, help="Number of CPU workers for DataLoader")

    args = parser.parse_args()

    train(args)