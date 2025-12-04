import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import pandas as pd
from utils.dataset import FeatureDataset
from model.fcn_classifier import FCN_Classifier
from utils.load_config import load_config
from utils.logger import get_logger

log = get_logger(__name__)

def train_fcn(train_features_df, val_features_df, config):
    """Training loop for the FCN on saved features."""

    # Retrieve necessary configuration variables
    DEVICE = config['model']['device']
    FEATURE_DIM = config['model']['feature_dim']
    NUM_EPOCHS = config['training']['epochs']
    BATCH_SIZE = config['training']['batch_size']
    LEARNING_RATE = config['training']['learning_rate']
    MODEL_WEIGHTS_PATH = config['paths']['model_weights_path']
    
    # 1. Setup Data Loaders
    train_dataset = FeatureDataset(train_features_df)
    val_dataset = FeatureDataset(val_features_df)
    
    # Use a large batch size (256) for fast training since features are small
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Setup Model, Loss, and Optimizer
    model = FCN_Classifier(FEATURE_DIM).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3. Training Loop
    best_val_loss = float('inf')
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        for features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * features.size(0)
        
        # 4. Validation
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(DEVICE), labels.to(DEVICE)
                outputs = model(features)
                val_loss += criterion(outputs, labels).item() * features.size(0)
                
                # Frame-level prediction: sigmoid converts logits to probability
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                correct += (predictions == labels).sum().item()

        avg_train_loss = total_loss / len(train_dataset)
        avg_val_loss = val_loss / len(val_dataset)
        val_accuracy = correct / len(val_dataset)
        
        log.info(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        
        # Save best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Save the trained model weights to Google Drive
            torch.save(model.state_dict(), MODEL_WEIGHTS_PATH)
            log.info("Model saved!")
            
    return model

if __name__ == "__main__":
    # Load features from CSV
    cfg = load_config()
    TRAIN_FEATURES_FILE = cfg["training"]["train_features_sub_dir"]
    VAL_FEATURES_FILE = cfg["training"]["val_features_sub_dir"]
    train_features_df = pd.read_csv(TRAIN_FEATURES_FILE)
    val_features_df = pd.read_csv(VAL_FEATURES_FILE)
    
    # Train the FCN model
    train_fcn(train_features_df, val_features_df, cfg)