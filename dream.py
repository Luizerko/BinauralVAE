import argparse
import os

import torch

from models.VAE import VAE
from models.CVAE import CVAE

from inference import reconstruction_mel, reconstruction_stft_4ch, reconstruction_stft_complex

# Helper function to create a trajectory through the latent space via interpolation
def generate_latent_trajectory(latent_dim, num_steps, num_keyframes, is_complex, device):
    # Sampling random keyframes from the prior distributions
    if is_complex:
        # Standard complex gaussian has variance 1 (0.5 real, 0.5 imag)
        keyframes = torch.randn(num_keyframes, latent_dim, device=device, dtype=torch.complex64)
    else:
        keyframes = torch.randn(num_keyframes, latent_dim, device=device)

    # Linearly interpolating between keyframes
    steps_per_segment = num_steps // (num_keyframes - 1)
    trajectory = []
    for i in range(num_keyframes - 1):
        start_key = keyframes[i]
        end_key = keyframes[i+1]
        
        alphas = torch.linspace(0, 1, steps_per_segment, device=device)
        for alpha in alphas[:-1]:
            trajectory.append(start_key * (1 - alpha) + end_key * alpha)
            
    # Adding the final keyframe and padding or trimming to hit exactly num_steps
    trajectory.append(keyframes[-1])
    while len(trajectory) < num_steps:
        trajectory.append(keyframes[-1])
    trajectory = trajectory[:num_steps]

    return torch.stack(trajectory)


# Dreaming loop
def dream(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Fetching a reference sample to get the correct input dimensions
    dataset_path = os.path.join(args.dataset_dir, f'seed_{args.reference_seed}', args.dataset_method)
    ref_sample = torch.load(os.path.join(dataset_path, 'binaural_image_1.pt'), map_location=device)
    sample_shape = ref_sample.shape
    
    # Initializing model and loading checkpoint
    latent_dim = 2 ** args.latent_dim_pow
    if args.dataset_method in ['mel', 'stft_4ch']:
        model = VAE(image_dimensions=(sample_shape[1], sample_shape[2]), image_channels=sample_shape[0], latent_dim_pow=args.latent_dim_pow, n_filters=args.n_filters, ks_v=args.kernel_v, ks_h=args.kernel_h, s_v=args.stride_v, s_h=args.stride_h, pad=args.pad).to(device)
    elif args.dataset_method == 'stft_complex':
        model = CVAE(image_dimensions=(sample_shape[1], sample_shape[2]), image_channels=sample_shape[0], latent_dim_pow=args.latent_dim_pow, n_filters=args.n_filters, ks_v=args.kernel_v, ks_h=args.kernel_h, s_v=args.stride_v, s_h=args.stride_h, pad=args.pad).to(device)

    checkpoint_path = os.path.join(args.checkpoint_path, args.dataset_method, args.run_name, 'model_save.pt')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Generating latent trajectory
    is_complex = (args.dataset_method == 'stft_complex')
    z_trajectory = generate_latent_trajectory(latent_dim, args.num_steps, args.num_keyframes, is_complex, device)

    # Decoding trajectory
    with torch.no_grad():
        # Using the explicit decoder branch matching your architecture
        if is_complex:
            reconstructions = model.decoder_input_1(z_trajectory)
            reconstructions = model.decoder_input_2(reconstructions)
            reconstructions = model.decoder(reconstructions)
        else:
            reconstructions = model.decoder_input(z_trajectory)
            reconstructions = model.decoder(reconstructions)
    reconstructed_data = reconstructions.cpu().numpy()

    # Reconstructing audio
    if args.dataset_method == 'mel':
        # Grabbing reference power from a reference seed folder
        ref_power = torch.load(os.path.join(args.dataset_dir, f"seed_{args.reference_seed}", 'mel_ref_power.pt'))['ref_power']
        reconstruction_mel(reconstructed_data, args.output_file, ref_power, args.n_samples, args.mel_bands, args.hop_len, args.sample_rate, args.rec_method)
    elif args.dataset_method == 'stft_4ch':
        reconstruction_stft_4ch(reconstructed_data, args.output_file, args.n_samples, args.hop_len, args.sample_rate)
    elif args.dataset_method == 'stft_complex':
        reconstruction_stft_complex(reconstructed_data, args.output_file, args.n_samples, args.hop_len, args.sample_rate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Parsing arguments
    parser.add_argument("--num_steps", type=int, default=32, help="Total number of frames to generate")
    parser.add_argument("--num_keyframes", type=int, default=2, help="Number of random latent points to interpolate between (greater than or equal to 2)")
    parser.add_argument("--reference_seed", type=int, required=True, help="Index of a real dataset seed (used purely to extract input shape dimensions and mel reference power)")
    
    parser.add_argument("--checkpoint_path", type=str, default='models/checkpoints/', help="Path to the saved model checkpoint")
    parser.add_argument("--run_name", type=str, default='test_run_0001', help="Run name to locate the model weights")
    parser.add_argument("--output_file", type=str, default='dream_output.wav', help="Path to the output file")
    parser.add_argument("--dataset_dir", type=str, default='data/dataset/', help="Path to processed .pt dataset directory")
    parser.add_argument("--dataset_method", type=str, default='mel', help="Method of audio processing used", choices=['mel', 'stft_4ch', 'stft_complex'])
    parser.add_argument("--rec_method", type=str, default='gl', help="Reconstruction method to be used (only for Mel spectrograms)", choices=['gl', 'bvg'])

    parser.add_argument("--latent_dim_pow", type=int, default=5, help="Size of the latent space (in powers of 2)")
    parser.add_argument("--n_filters", type=int, default=3, help="Number of conv layers")
    parser.add_argument("--kernel_v", type=int, default=7, help="Kernel size vertically")
    parser.add_argument("--kernel_h", type=int, default=5, help="Kernel size horizontally")
    parser.add_argument("--stride_v", type=int, default=2, help="Vertical stride")
    parser.add_argument("--stride_h", type=int, default=1, help="Horizontal stride")
    parser.add_argument("--pad", type=int, default=0, help="Amount of padding")

    parser.add_argument("--n_samples", type=int, default=2048, help="Amount of samples used for Mel/STFT processing.")
    parser.add_argument("--mel_bands", type=int, default=128, help="Amount of Mel bands used.")
    parser.add_argument("--hop_len", type=int, default=147, help="Time resolution used for Mel/STFT processing.")
    parser.add_argument("--sample_rate", type=int, default=44100, help="Sample rate of original audio.")

    args = parser.parse_args()

    dream(args)