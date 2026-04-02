import sys

sys.path.insert(0, '/media/aaa/d6249A89249A6BED/anaconda3/envs/mamba310/lib/python3.10/site-packages')

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import scipy.io as sio


class PreprocessedEEGDataset(Dataset):

    def __init__(self, data_file: str):

        print(f"Loading data from {data_file}...")

        mat_data = sio.loadmat(data_file)

        self.clean_data = mat_data['clean_data']
        self.noisy_data = mat_data['noisy_data']
        self.labels = mat_data['labels'].flatten()

        if 'trial_info' in mat_data:
            self.trial_info = mat_data['trial_info'].flatten()
        else:
            self.trial_info = None

        self.snr = mat_data.get('snr', None)
        if self.snr is not None:
            self.snr = float(self.snr)

        print(f"  Loaded - Clean: {self.clean_data.shape}, Noisy: {self.noisy_data.shape}, Labels: {self.labels.shape}")
        print(f"  SNR: {self.snr}dB")
        print(f"  Label distribution: {np.bincount(self.labels)}")

    def __getitem__(self, idx):
        clean = torch.FloatTensor(self.clean_data[idx])
        noisy = torch.FloatTensor(self.noisy_data[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return clean, noisy, label

    def __len__(self):
        return len(self.clean_data)

