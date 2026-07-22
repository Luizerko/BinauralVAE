import argparse
import os
import shutil
from pathlib import Path

import torch
from torch.utils.data import Dataset

import numpy as np

# Creating generic torch dataset for btaching and paralelization later on
class BinauralDataset(Dataset):
    def __init__(self, dataset_dir, method):
        self.file_paths = []

        # Iterating through the dataset directories and collecting all binaural images as one big dataset
        seeds = [seed for seed in Path(dataset_dir).iterdir() if seed.is_dir()]
        for seed in sorted(seeds, key=lambda x: x.name):
            seed_method = seed / method
            binaural_images = sorted(seed_method.glob('*.pt'), key=lambda x: int(x.stem.split('_')[2]))
            self.file_paths.extend(binaural_images)

    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        return torch.load(self.file_paths[idx])


# Creating and saving Mel dataset
def dataset_create_mel(src_dir, tgt_dir, num_seeds):
    # Computing normalization statistics
    global_min = float('inf')
    global_max = float('-inf')
    for seed in range(1, num_seeds+1):
        # Excepting broken simulation runs
        try:
            src_file = os.path.join(src_dir, f'seed_{seed}', 'split_mel.npz')
            data = np.load(src_file)
            mel_left = data['left']
            mel_right = data['right']
            
            seed_min = min(mel_left.min(), mel_right.min())
            seed_max = max(mel_left.max(), mel_right.max())
            
            if seed_min < global_min:
                global_min = seed_min
            if seed_max > global_max:
                global_max = seed_max

            # Also saving reference power for good reconstruction later on
            ref_power_path = os.path.join(tgt_dir, f'seed_{seed}')
            os.makedirs(ref_power_path, exist_ok=True)
            ref_power = np.load(os.path.join(src_dir, f'seed_{seed}', 'mel_ref_power.npz'))['ref_power']

            # Fixing a folder creation mistake on an older version of this script
            ref_power_pt_path = os.path.join(ref_power_path, 'mel_ref_power.pt')
            if os.path.isdir(ref_power_pt_path):
                shutil.rmtree(ref_power_pt_path)

            torch.save({'ref_power': float(ref_power)}, ref_power_pt_path)
        except:
            continue

    # Saving normalization statistics for later audio reconstruction
    stats = {'min': float(global_min), 'max': float(global_max)}
    torch.save(stats, 'mel_stats.pt')
    
    # Normalizing and saving chunks
    for seed in range(1, num_seeds+1):
        # Excepting broken simulation runs
        try:
            # Configuring files and directories
            src_file = os.path.join(src_dir, f'seed_{seed}', 'split_mel.npz')
            seed_tgt_dir = os.path.join(tgt_dir, f'seed_{seed}/mel')
            os.makedirs(seed_tgt_dir, exist_ok=True)

            # Loading the data
            data = np.load(src_file)
            mel_left = data['left']
            mel_right = data['right']

            # Iterating through the data to save binaural image
            num_chunks = mel_left.shape[0]
            for i in range(num_chunks):
                chunk_left = mel_left[i]
                chunk_right = mel_right[i]

                chunk_binaural = np.stack((chunk_left, chunk_right), axis=0)
                chunk_binaural = (chunk_binaural - global_min) / (global_max - global_min + 1e-8)
                chunk_binaural = torch.tensor(chunk_binaural, dtype=torch.float32)
                
                torch.save(chunk_binaural, os.path.join(seed_tgt_dir, f'binaural_image_{i+1}.pt'))
        except:
            continue

# Creating and saving STFT 4-channel dataset
def dataset_create_stft_4ch(src_dir, tgt_dir, num_seeds):
    # Computing normalization statistics
    global_mag_min = float('inf')
    global_mag_max = float('-inf')
    for seed in range(1, num_seeds+1):
        # Excepting broken simulation runs
        try:
            src_file = os.path.join(src_dir, f'seed_{seed}', 'split_stft.npz')
            data = np.load(src_file)
            stft_left = data['left']
            stft_right = data['right']
            
            seed_mag_min = min(np.abs(stft_left).min(), np.abs(stft_right).min())
            seed_mag_max = max(np.abs(stft_left).max(), np.abs(stft_right).max())
            
            if seed_mag_min < global_mag_min:
                global_mag_min = seed_mag_min
            if seed_mag_max > global_mag_max:
                global_mag_max = seed_mag_max
        except:
            continue

    # Saving normalization statistics for later audio reconstruction
    stats = {'mag_min': float(global_mag_min), 'mag_max': float(global_mag_max), 'phase_min': float(-np.pi), 'phase_max': float(np.pi)}
    torch.save(stats, 'stft_stats.pt')

    for seed in range(1, num_seeds+1):
        # Excepting broken simulation runs
        try:
            # Configuring files and directories
            src_file = os.path.join(src_dir, f'seed_{seed}', 'split_stft.npz')
            seed_tgt_dir = os.path.join(tgt_dir, f'seed_{seed}/stft_4ch')
            os.makedirs(seed_tgt_dir, exist_ok=True)

            # Loading the data
            data = np.load(src_file)
            stft_left = data['left']
            stft_right = data['right']
            mag_left, phase_left = np.abs(stft_left), np.angle(stft_left)
            mag_right, phase_right = np.abs(stft_right), np.angle(stft_right)

            # Iterating through the data to save binaural "image"
            num_chunks = stft_left.shape[0]
            for i in range(num_chunks):
                chunk_mag_left = (mag_left[i] - global_mag_min)/(global_mag_max - global_mag_min + 1e-8)
                chunk_mag_right = (mag_right[i] - global_mag_min)/(global_mag_max - global_mag_min + 1e-8)
                
                chunk_mag_left = chunk_mag_left ** 0.3
                chunk_mag_right = chunk_mag_right ** 0.3

                chunk_phase_left = (phase_left[i] + np.pi)/(2 *np.pi)
                chunk_phase_right = (phase_right[i] + np.pi)/(2 *np.pi)

                chunk_binaural = np.stack((chunk_mag_left, chunk_phase_left, chunk_mag_right, chunk_phase_right), axis=0)
                chunk_binaural = torch.tensor(chunk_binaural, dtype=torch.float32)
                
                torch.save(chunk_binaural, os.path.join(seed_tgt_dir, f'binaural_image_{i+1}.pt'))
        except:
            continue


# Creating and saving STFT complex dataset
def dataset_create_stft_complex(src_dir, tgt_dir, num_seeds):
    # Computing normalization statistics
    global_mag_max = float('-inf')
    for seed in range(1, num_seeds+1):
        # Excepting broken simulation runs
        try:
            src_file = os.path.join(src_dir, f'seed_{seed}', 'split_stft.npz')
            data = np.load(src_file)
            stft_left = data['left']
            stft_right = data['right']
            
            seed_mag_max = max(np.abs(stft_left).max(), np.abs(stft_right).max())
            
            if seed_mag_max > global_mag_max:
                global_mag_max = seed_mag_max
        except:
            continue

    # Saving normalization statistics for later audio reconstruction
    stats = {'mag_max': float(global_mag_max)}
    torch.save(stats, 'complex_stats.pt')
    
    for seed in range(1, num_seeds+1):
        # Excepting broken simulation runs
        try:
            # Configuring files and directories
            src_file = os.path.join(src_dir, f'seed_{seed}', 'split_stft.npz')
            seed_tgt_dir = os.path.join(tgt_dir, f'seed_{seed}/stft_complex')
            os.makedirs(seed_tgt_dir, exist_ok=True)

            # Loading the data
            data = np.load(src_file)
            stft_left = data['left']
            stft_right = data['right']

            # Iterating through the data to save complex binaural "image"
            num_chunks = stft_left.shape[0]
            for i in range(num_chunks):
                norm_left = stft_left[i]/global_mag_max
                norm_right = stft_right[i]/global_mag_max

                mag_l, phase_l = np.abs(norm_left), np.angle(norm_left)
                mag_r, phase_r = np.abs(norm_right), np.angle(norm_right)

                mag_l = mag_l ** 0.3
                mag_r = mag_r ** 0.3

                chunk_left = mag_l * np.exp(1j * phase_l)
                chunk_right = mag_r * np.exp(1j * phase_r) 

                chunk_binaural = np.stack((chunk_left, chunk_right), axis=0)
                chunk_binaural = torch.tensor(chunk_binaural, dtype=torch.complex64)
                torch.save(chunk_binaural, os.path.join(seed_tgt_dir, f'binaural_image_{i+1}.pt'))
        except:
            continue


if __name__ == '__main__':
    # Parsing arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--source_dir", help="Path to where your raw data is located.", type=str, default='/mnt/f/lz/visgraf/soundspaces/data/mp3d/alarm/17DRP5sb8fy/rollout/target/')
    parser.add_argument("--target_dir", help="Path to where you want to save your processed data.", type=str, default='dataset/')
    parser.add_argument("--num_seeds", help="Number of seeds to process", type=int, default=1500)

    args = parser.parse_args()

    # Calling the appropriate method for dataset creation and saving
    dataset_create_mel(args.source_dir, args.target_dir, args.num_seeds)
    dataset_create_stft_4ch(args.source_dir, args.target_dir, args.num_seeds)
    dataset_create_stft_complex(args.source_dir, args.target_dir, args.num_seeds)