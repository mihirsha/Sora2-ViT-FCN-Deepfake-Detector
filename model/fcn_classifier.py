import torch.nn as nn
from utils.load_config import load_config

config = load_config()
DEVICE = config['model']['DEVICE']
FEATURE_DIM = int(config["model"]["feature_dim"])

class FCN_Classifier(nn.Module):
    """Simple Fully Connected Network (MLP) for binary classification of ViT features."""
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.layers(x)