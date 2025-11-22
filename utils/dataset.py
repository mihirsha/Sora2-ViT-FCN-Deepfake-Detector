import numpy as np
import torch
from torch.utils.data import Dataset

class FeatureDataset(Dataset):
    """Custom Dataset for loading pre-extracted features."""
    def __init__(self, features_df):
        self.features = features_df.drop('label', axis=1).values.astype(np.float32)
        self.labels = features_df['label'].values.astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx]), torch.tensor(self.labels[idx]).unsqueeze(0)