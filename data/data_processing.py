import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset

import numpy as np

# Creating generic torch dataset for btaching and paralelization later on
class BinauralDataset(Dataset):
    def __init__(self, dataset_dir):
        self.file_paths = []

        # Iterating through the dataset directories and collecting all binaural images as one big dataset
        for seed in sorted(Path(dataset_dir).iterdir(), key=lambda x: x.name):
            binaural_images = sorted(seed.glob('*.pt'), key=lambda x: int(x.stem.split('_')[1]))
            self.file_paths.extend(binaural_images)

    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        return torch.load(self.file_paths[idx])

# Creating and saving the dataset
def dataset_create_mel(src_dir, tgt_dir, num_seeds):
    for seed in range(1, num_seeds+1):
        # Configuring files and directories
        src_file = os.path.join(src_dir, f'seed{seed}', 'split_mel.npz')
        tgt_dir = os.path.join(tgt_dir, f'seed_{seed}')
        os.makedirs(tgt_dir, exist_ok=True)

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
            chunk_binaural = torch.tensor(chunk_binaural, dtype=torch.float32)
            torch.save(chunk_binaural, os.path.join(tgt_dir, f'binaural_image_{i+1}.pt'))

if __name__ == '__main__':
    # Parsing arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--source_dir", help="Path to where your raw data is located.", type=str, default='/mnt/c/Users/luisvz/Documents/visgraf/soundspaces/data/mp3d_example/sounds/alarm/simulation/target/')
    parser.add_argument("--target_dir", help="Path to where you want to save your processed data.", type=str, default='dataset/')
    
    parser.add_argument("--method", help="Choosing a method to process the audio.", type=str, default='mel', choices=['mel', 'stft', 'wav2vec'])
    parser.add_argument("--num_seeds", help="Number of seeds to process", type=int, default=1000)

    args = parser.parse_args()

    # Calling the appropriate method for dataset creation and saving    
    if args.method == 'mel':
        dataset_create_mel(args.source_dir, args.target_dir, args.num_seeds)