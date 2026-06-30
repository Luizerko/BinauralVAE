import argparse
import os

import torch
import matplotlib.pyplot as plt

from data.data_processing import BinauralDataset
from models.VAE import VAE


# Inference and plotting function
def infer_and_plot(args):
    # Setting up device and dataset
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = BinauralDataset(args.dataset_dir, args.dataset_method)
    
    if args.dataset_index >= len(dataset) or args.dataset_index < 0:
        print(f'Dataset index out of bound. Dataset size: {len(dataset)}')
        return

    # Fetching the original sample
    original_sample = dataset[args.dataset_index]
    sample_shape = original_sample.shape
    
    # Initializing the model
    if args.dataset_method in ['mel', 'stft_4ch']:
        model = VAE(image_dimensions=(sample_shape[1], sample_shape[2]), image_channels=sample_shape[0], latent_dim_pow=args.latent_dim_pow, n_filters=args.n_filters, ks_v=args.kernel_v, ks_h=args.kernel_h, s_v=args.stride_v, s_h=args.stride_h, pad=args.pad).to(device)

    # Loading checkpoint
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Running inference
    x = original_sample.unsqueeze(0).to(device)
    with torch.no_grad():
        reconstruction, mu, logvar = model(x)
        
    # Removing batch dimension and moving back to CPU for plotting
    original_data = original_sample.cpu().numpy()
    reconstructed_data = reconstruction.squeeze(0).cpu().numpy()

    plot_reconstruction(original_data, reconstructed_data)


# Plotting function
def plot_reconstruction(original, reconstruction):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    fig.suptitle('Spatial Audio Reconstruction', fontsize=16)
 
    # Left ears
    ax_orig_l = axes[0, 0]
    im1 = ax_orig_l.imshow(original[0], aspect='auto', origin='lower', cmap='viridis')
    ax_orig_l.set_title("Original - Left Ear")
    fig.colorbar(im1, ax=ax_orig_l)

    ax_rec_l = axes[0, 1]
    im2 = ax_rec_l.imshow(reconstruction[0], aspect='auto', origin='lower', cmap='viridis')
    ax_rec_l.set_title("Reconstruction - Left Ear")
    fig.colorbar(im2, ax=ax_rec_l)

    # Right ears
    ax_orig_r = axes[1, 0]
    im3 = ax_orig_r.imshow(original[1], aspect='auto', origin='lower', cmap='viridis')
    ax_orig_r.set_title("Original - Right Ear")
    fig.colorbar(im3, ax=ax_orig_r)

    ax_rec_r = axes[1, 1]
    im4 = ax_rec_r.imshow(reconstruction[1], aspect='auto', origin='lower', cmap='viridis')
    ax_rec_r.set_title("Reconstruction - Right Ear")
    fig.colorbar(im4, ax=ax_rec_r)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Parsing arguments
    parser.add_argument("--checkpoint_path", type=str, default='models/checkpoints/model_save.pt', help="Path to the saved model checkpoint")
    parser.add_argument("--dataset_index", type=int, required=True, help="Index of the sample in the dataset to reconstruct")

    parser.add_argument("--dataset_dir", type=str, default='data/dataset/', help="Path to processed .pt dataset directory")
    parser.add_argument("--dataset_method", type=str, default='mel', help="Method of audio processing used", choices=['mel', 'stft_4ch', 'stft_complex', 'wave2vec'])
    
    parser.add_argument("--latent_dim_pow", type=int, default=5, help="Size of the latent space (in powers of 2)")
    parser.add_argument("--n_filters", type=int, default=3, help="Number of conv layers")
    parser.add_argument("--kernel_v", type=int, default=7, help="Kernel size vertically")
    parser.add_argument("--kernel_h", type=int, default=5, help="Kernel size horizontally")
    parser.add_argument("--stride_v", type=int, default=2, help="Vertical stride")
    parser.add_argument("--stride_h", type=int, default=1, help="Horizontal stride")
    parser.add_argument("--pad", type=int, default=0, help="Amount of padding")

    args = parser.parse_args()

    infer_and_plot(args)