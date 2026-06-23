import os
import argparse
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data.data_processing import BinauralDataset
from models.VAE import VAE
# from models.CVAE import CVAE

# Main training loop
def train(args):
    # Setting up device and dataset/dataloader
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = BinauralDataset(args.dataset_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # Initializing model, optimizer and logger
    sample_shape = dataset[0].shape
    if args.dataset_method == 'mel' or args.dataset_method == 'stft_4ch':
        model = VAE(image_dimensions=(sample_shape[1], sample_shape[2]), image_channels=sample_shape[0], latent_dim=args.latent_dim).to(device)
    model.apply(model.init_weights(method='he'))

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    writer = SummaryWriter(log_dir=args.log_dir)

    # Training loop
    model.train()
    best_loss = float('inf')
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
                'Rec Loss': f'{rec_loss.item():.4f}',
                'KL Loss': f'{kl_loss.item():.4f}'
            })

        avg_rec_loss = epoch_rec_loss/len(dataloader)
        avg_kl_loss = epoch_kl_loss/len(dataloader)
        avg_total_loss = epoch_total_loss/len(dataloader)
        print(f"Epoch {epoch} average loss: {avg_total_loss:.4f}\n")

        # Logging to tensorboard
        writer.add_scalar('Loss/Reconstruction', avg_rec_loss, epoch)
        writer.add_scalar('Loss/KL_Divergence', avg_kl_loss, epoch)
        writer.add_scalar('Loss/Total', avg_total_loss, epoch)

        for name, param in model.named_parameters():
            writer.add_histogram(f'Weights/{name}', param, epoch)
            if param.grad is not None:
                writer.add_histogram(f'Gradients/{name}', param.grad, epoch)

        if avg_total_loss < best_loss:
            best_loss = avg_total_loss
            checkpoint_path = os.path.join(args.save_dir, f'model_save.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_total_loss,
            }, checkpoint_path)

    writer.close()


if __name__ == '__main__':
    # Parsing arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_dir", type=str, default='data/dataset/', help="Path to processed .pt dataset directory", required=True)
    parser.add_argument("--dataset_method", type=str, default='mel', help="Method of audio processing used for the dataset creation", choices=['mel', 'stft_4ch', 'stft_complex', 'wave2vec'], required=True)
    parser.add_argument("--save_dir", type=str, default='models/checkpoints/', help="Directory to save model weights")
    parser.add_argument("--log_dir", type=str, default='models/logs/', help="Directory for logging during model training")

    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for optimizer")
    parser.add_argument("--latent_dim", type=int, default=32, help="Size of the latent space")
    parser.add_argument("--beta", type=float, default=2.0, help="Beta weight for KL Divergence loss")

    parser.add_argument("--num_workers", type=int, default=4, help="Number of CPU workers for DataLoader")

    args = parser.parse_args()

    args.dataset_dir += args.dataset_method + '/'
    args.save_dir += args.dataset_method + '/'
    args.log_dir += args.dataset_method + '/'

    train(args)