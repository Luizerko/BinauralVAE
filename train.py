import os
import argparse
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from data.data_processing import BinauralDataset
from models.VAE import VAE
# from models.CVAE import CVAE

# Main training loop
def train(args):
    # Setting up device and dataset/dataloader
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = BinauralDataset(args.dataset_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # Initializing model and optimizer
    sample_shape = dataset[0].shape
    model = VAE(image_dimensions=(sample_shape[1], sample_shape[2]), image_channels=sample_shape[0], latent_dim=args.latent_dim).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Training loop
    model.train()
    for epoch in range(1, args.epochs+1):
        epoch_rec_loss = 0.0
        epoch_kl_loss = 0.0
        epoch_total_loss = 0.0

        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Epoch {epoch}/{args.epochs}')
        for batch_idx, batch in progress_bar:
            # Forward pass
            x = batch.to(device)

            optimizer.zero_grad()
            rec, mu, logvar = model(x)

            # Loss computation and backward pass
            rec_loss, kl_loss, total_loss = model.loss(rec, x, mu, logvar, beta=args.beta)
            
            total_loss.backward()
            optimizer.step()

            epoch_rec_loss += rec_loss.item()
            epoch_kl_loss += kl_loss.item()
            epoch_total_loss += total_loss.item()

            # Logging
            progress_bar.set_postfix({
                'Total Loss': f'{total_loss.item():.4f}',
                'Rec Loss': f'{total_loss.item():.4f}',
                'KL Loss': f'{kl_loss.item():.4f}'
            })

        avg_total_loss = epoch_total_loss/len(dataloader)
        print(f"Epoch {epoch} average loss: {avg_total_loss:.4f}\n")

        checkpoint_path = os.path.join(args.save_dir, f'vae_epoch_{epoch}.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_total_loss,
        }, checkpoint_path)

if __name__ == '__main__':
    # Parsing arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_dir", type=str, default='data/dataset/', help="Path to processed .pt dataset directory")
    parser.add_argument("--save_dir", type=str, default='models/checkpoints/mel/', help="Directory to save model weights")

    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for optimizer")
    parser.add_argument("--latent_dim", type=int, default=32, help="Size of the latent space")
    parser.add_argument("--beta", type=float, default=2.0, help="Beta weight for KL Divergence loss")

    parser.add_argument("--num_workers", type=int, default=4, help="Number of CPU workers for DataLoader")

    args = parser.parse_args()

    train(args)