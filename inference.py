import argparse
import os
import sys

import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from data.data_processing import BinauralDataset
from models.VAE import VAE


# Inference and plotting function
def infer_and_plot(args):
    # Setting up device and dataset
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = BinauralDataset(args.dataset_dir, args.dataset_method)
    
    if args.dataset_idx_ini >= len(dataset) or args.dataset_idx_ini < 0 or args.dataset_idx_end >= len(dataset) or args.dataset_idx_end < 0:
        print(f'Dataset index out of bound. Dataset size: {len(dataset)}')
        return

    # Fetching the original sample
    original_samples = torch.stack([dataset[i] for i in range(args.dataset_idx_ini, args.dataset_idx_end)])
    sample_shape = original_samples[0].shape
    
    # Initializing the model
    if args.dataset_method in ['mel', 'stft_4ch']:
        model = VAE(image_dimensions=(sample_shape[1], sample_shape[2]), image_channels=sample_shape[0], latent_dim_pow=args.latent_dim_pow, n_filters=args.n_filters, ks_v=args.kernel_v, ks_h=args.kernel_h, s_v=args.stride_v, s_h=args.stride_h, pad=args.pad).to(device)

    # Loading checkpoint
    checkpoint_path = os.path.join(args.checkpoint_path, args.dataset_method, args.run_name, 'model_save.pt')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Running inference
    x = original_samples.to(device)
    with torch.no_grad():
        reconstructions, _, _ = model(x)
        
    # Moving back to CPU for plotting
    original_data = original_samples.cpu().numpy()
    reconstructed_data = reconstructions.cpu().numpy()

    # Plotting reconstruction and then actually reconstructing audio
    if args.dataset_method == 'mel':
        plot_mel(original_data[0], reconstructed_data[0])
        reconstruction_mel(original_data, 'original.wav', args.n_samples, args.mel_bands, args.hop_len, args.sample_rate, args.rec_method)
        reconstruction_mel(reconstructed_data, args.output_file, args.n_samples, args.mel_bands, args.hop_len, args.sample_rate, args.rec_method)

    elif args.dataset_method == 'stft_4ch':
        plot_stft_4ch(original_data[0], reconstructed_data[0])
        reconstruction_stft_4ch(original_data, 'original.wav', args.n_samples, args.hop_len, args.sample_rate)
        reconstruction_stft_4ch(reconstructed_data, args.output_file, args.n_samples, args.hop_len, args.sample_rate)

    return original_data, reconstructed_data


# Plotting Mel reconstruction
def plot_mel(original, reconstruction):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    fig.suptitle('Spatial Audio Reconstruction', fontsize=16)
 
    # Left ears
    ax_orig_l = axes[0, 0]
    im1 = ax_orig_l.imshow(original[0], aspect='auto', origin='lower', cmap='magma')
    ax_orig_l.set_title("Original - Left Ear")
    fig.colorbar(im1, ax=ax_orig_l)

    ax_rec_l = axes[0, 1]
    im2 = ax_rec_l.imshow(reconstruction[0], aspect='auto', origin='lower', cmap='magma')
    ax_rec_l.set_title("Reconstruction - Left Ear")
    fig.colorbar(im2, ax=ax_rec_l)

    # Right ears
    ax_orig_r = axes[1, 0]
    im3 = ax_orig_r.imshow(original[1], aspect='auto', origin='lower', cmap='magma')
    ax_orig_r.set_title("Original - Right Ear")
    fig.colorbar(im3, ax=ax_orig_r)

    ax_rec_r = axes[1, 1]
    im4 = ax_rec_r.imshow(reconstruction[1], aspect='auto', origin='lower', cmap='magma')
    ax_rec_r.set_title("Reconstruction - Right Ear")
    fig.colorbar(im4, ax=ax_rec_r)

    plt.tight_layout()
    plt.show()
    plt.close()


# Plotting STFT-4ch reconstruction
def plot_stft_4ch(original, reconstruction):
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(12, 16))
    fig.suptitle('STFT 4-Channel Spatial Audio Reconstruction', fontsize=16)

    channel_names = ["Left Ear Magnitude", "Left Ear Phase", "Right Ear Magnitude", "Right Ear Phase"]
    cmaps = ['magma', 'twilight', 'magma', 'twilight']

    for i in range(4):
        # Original column
        ax_orig = axes[i, 0]
        im_orig = ax_orig.imshow(original[i], aspect='auto', origin='lower', cmap=cmaps[i])
        ax_orig.set_title(f"Original: {channel_names[i]}")
        fig.colorbar(im_orig, ax=ax_orig)

        # Reconstruction column
        ax_rec = axes[i, 1]
        im_rec = ax_rec.imshow(reconstruction[i], aspect='auto', origin='lower', cmap=cmaps[i])
        ax_rec.set_title(f"Reconstruction: {channel_names[i]}")
        fig.colorbar(im_rec, ax=ax_rec)

    plt.tight_layout()
    plt.show()
    plt.close()


# Reconstructing audio from Mel model's output
def reconstruction_mel(data, output_file='output_mel.wav', n_fft=2048, n_mels=128, hop_len=147, sample_rate=44100, rec_method='GriffinLim'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mel_tensor = torch.FloatTensor(data).to(device)

    # Unormalizing data
    stats = torch.load('data/mel_stats.pt', map_location=device)
    mel_min, mel_max = stats['min'], stats['max']
    mel_tensor = (mel_tensor * (mel_max - mel_min + 1e-8)) + mel_min

    # Converting dB to power and then to magnitude
    ref_power = 10000.0 
    mel_tensor = ref_power * (10.0 ** (mel_tensor / 10.0))
    mel_tensor = torch.sqrt(mel_tensor)

    if rec_method == 'BigVGAN':
        # Loading BigVGAN-v2 model to map mel spectrogram back to audio. The used checkpoint used hop length of 512 instead of our 147, so the audio is probably going to sound slow
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.append(os.path.join(current_dir, 'bigvgan'))
        
        import bigvgan
        model = bigvgan.BigVGAN.from_pretrained('nvidia/bigvgan_v2_44khz_128band_256x', use_cuda_kernel=False).to(device)
        model.remove_weight_norm()
        model.eval()

        # Processing spectrograms separately since vocoders are normally designed for mono audio, not stereo, including BigVGAN-v2
        mel_tensor = torch.log(torch.clamp(mel_tensor, min=1e-5))
        with torch.no_grad():
            wav_l = model(mel_tensor[:, 0]).cpu().numpy()
            wav_r = model(mel_tensor[:, 1]).cpu().numpy()
    
    elif rec_method == 'GriffinLim':
        # Converiting Mel spectrogram to (magnitude) SFTF and then running Griffim-Lim method
        inverse_mel = torchaudio.transforms.InverseMelScale(n_stft=n_fft//2 + 1, n_mels=n_mels, sample_rate=sample_rate).to(device)
        griffin_lim = torchaudio.transforms.GriffinLim(n_fft=n_fft, hop_length=hop_len, power=1.0).to(device)
    
        with torch.no_grad():
            linear_stft_l = inverse_mel(mel_tensor[:, 0])
            wav_l = griffin_lim(linear_stft_l).cpu().numpy()
            
            linear_stft_r = inverse_mel(mel_tensor[:, 1])
            wav_r = griffin_lim(linear_stft_r).cpu().numpy()

    # Peak-normalization and audio saving
    wav_l = wav_l.flatten()
    wav_r = wav_r.flatten()
    stereo_wav = np.stack([wav_l, wav_r], axis=1)
    max_amplitude = np.max(np.abs(stereo_wav))
    if max_amplitude > 0:
        stereo_wav = stereo_wav / (max_amplitude + 1e-9)
    sf.write(output_file, stereo_wav, sample_rate)


# Reconstructing audio from STFT-4ch model's output
def reconstruction_stft_4ch(data, output_file='output_stft.wav', n_fft=1024, hop_length=147, sample_rate=44100):
    # Converting to tensors
    mag_l = torch.from_numpy(data[:, 0])
    phase_l = torch.from_numpy(data[:, 1])
    mag_r = torch.from_numpy(data[:, 2])
    phase_r = torch.from_numpy(data[:, 3])

    # Unormalizing data
    stats = torch.load('data/stft_stats.pt')
    mag_min, mag_max = stats['mag_min'], stats['mag_max']
    
    mag_l = (mag_l * (mag_max - mag_min + 1e-8)) + mag_min
    mag_r = (mag_r * (mag_max - mag_min + 1e-8)) + mag_min

    phase_l = (phase_l * (2 * np.pi)) - np.pi
    phase_r = (phase_r * (2 * np.pi)) - np.pi

    # Mapping magnitude and angle back to complex numbers using Euler's formula (z = m * e^(i * phi))
    complex_l = mag_l * torch.exp(1j * phase_l)
    complex_r = mag_r * torch.exp(1j * phase_r)

    # Inverse STFT with the same window, sample and hop length as the forward process 
    window = torch.hann_window(n_fft)
    wav_l = torch.istft(complex_l, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=False)
    wav_r = torch.istft(complex_r, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=False)

    # Stacking stereo in [Time, Channels] and saving
    wav_l = wav_l.flatten().numpy()
    wav_r = wav_r.flatten().numpy()
    stereo_wav = np.stack([wav_l, wav_r], axis=1)
    sf.write(output_file, stereo_wav, sample_rate) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Parsing arguments
    parser.add_argument("--checkpoint_path", type=str, default='models/checkpoints/', help="Path to the saved model checkpoint")
    parser.add_argument("--run_name", type=str, default='test_run_0001', help="Run name for proper Tensorboard visualization")
    parser.add_argument("--output_file", type=str, default='reconstructed.wav', help="Path to the output file")
    parser.add_argument("--dataset_dir", type=str, default='data/dataset/', help="Path to processed .pt dataset directory")

    parser.add_argument("--dataset_idx_ini", type=int, required=True, help="Index of initial sample in the dataset to reconstruct")
    parser.add_argument("--dataset_idx_end", type=int, required=True, help="Index of final sample in the dataset to reconstruct")
    parser.add_argument("--dataset_method", type=str, default='mel', help="Method of audio processing used", choices=['mel', 'stft_4ch', 'stft_complex', 'wave2vec'])
    parser.add_argument("--rec_method", type=str, default='GriffinLim', help="Reconstruction method to be used (only for Mel spectrograms)", choices=['GriffinLim', 'BigVGAN'])

    parser.add_argument("--latent_dim_pow", type=int, default=5, help="Size of the latent space (in powers of 2)")
    parser.add_argument("--n_filters", type=int, default=3, help="Number of conv layers")
    parser.add_argument("--kernel_v", type=int, default=7, help="Kernel size vertically")
    parser.add_argument("--kernel_h", type=int, default=5, help="Kernel size horizontally")
    parser.add_argument("--stride_v", type=int, default=2, help="Vertical stride")
    parser.add_argument("--stride_h", type=int, default=1, help="Horizontal stride")
    parser.add_argument("--pad", type=int, default=0, help="Amount of padding")

    parser.add_argument("--n_samples", type=int, default=2048, help="Amount of samples used for Mel/STFT processing. Important metric for reconstruction")
    parser.add_argument("--mel_bands", type=int, default=128, help="Amount of Mel bands used. Important metric for Mel reconstruction")
    parser.add_argument("--hop_len", type=int, default=147, help="Time resolution used for Mel/STFT processing. Important metric for reconstruction")
    parser.add_argument("--sample_rate", type=int, default=44100, help="Sample rate of original audio. Important for reconstruction")

    args = parser.parse_args()

    infer_and_plot(args)