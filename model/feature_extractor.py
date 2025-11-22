from torchvision import models
from tqdm import tqdm
from utils.load_config import load_config
from utils.logger import get_logger
import torch.nn as nn
import torch
import pandas as pd
from pathlib import Path
from utils.video_processing import VideoFrameExtractor

log = get_logger(__name__)

config = load_config()
DEVICE = config['model']['DEVICE']
FEATURE_OUTPUT_FILE = config["training"]["feature_output_file"]
FEATURE_DIM = int(config["model"]["feature_dim"])
PATH_TO_FAKE_VIDEOS = config["training"]["fake_videos_sub_dir"]
PATH_TO_REAL_VIDEOS = config["training"]["real_videos_sub_dir"]

def setup_feature_extractor():
    """Loads a pre-trained ViT-Base and removes the classification head."""
    # Use the best available weights
    weights = models.ViT_B_16_Weights.IMAGENET1K_V1
    model = models.vit_b_16(weights=weights)

    # Freeze the weights for efficiency and transfer learning
    for param in model.parameters():
        param.requires_grad = False

    # In torchvision ViT, removing the classification layer.
    model.head = nn.Identity()
    model.eval()
    model.to(DEVICE)

    # Get the official preprocessing transforms
    preprocess = weights.transforms()

    log.info(f"ViT loaded and frozen. Feature dimension: {FEATURE_DIM}")
    return model, preprocess


def extract_and_save_features(
        model, 
        preprocess_func, 
        feature_output_path=FEATURE_OUTPUT_FILE, 
        path_to_fake_videos=PATH_TO_FAKE_VIDEOS, 
        path_to_real_videos=PATH_TO_REAL_VIDEOS
    ):
    """Main function to run the two-pass feature extraction."""

    extractor = VideoFrameExtractor(preprocess_func, sample_rate_fps=10) # 10 FPS target for max data
    all_data = [] # List to store {feature: vector, label: 0/1}

    # Check the path objects and glob results
    fake_video_glob = list(Path(path_to_fake_videos).glob("*.mp4"))
    real_video_glob = list(Path(path_to_real_videos).glob("*.mp4"))

    log.info(f"Fake Videos Dir: {Path(path_to_fake_videos)}")
    log.info(f"Real Videos Dir: {Path(path_to_real_videos)}")
    log.info(f"Found {len(fake_video_glob)} fake video files.")
    log.info(f"Found {len(real_video_glob)} real video files.")

    # Loop over all video files (adjust paths as needed)
    video_paths = fake_video_glob + real_video_glob

    if not video_paths:
        log.error("No video files found. Please ensure your PATH_TO_... variables are correct and the videos are present.")
        return

    for i, video_path in enumerate(tqdm(video_paths, desc="Extracting Features")):
        # Check the label based on the parent directory name or a reliable name component
        label = 1 if 'fake' in str(video_path).lower() else 0 # Using str(video_path) for robustness

        # Extract and preprocess frames
        frames_tensors, labels = extractor.extract(video_path, label)

        if not frames_tensors:
            continue

        # Combine all frame tensors into a single batch
        # This is where VRAM is utilized, but only for the frames from one video at a time.
        frame_batch = torch.stack(frames_tensors).to(DEVICE)

        with torch.no_grad():
            # Run forward pass through the ViT backbone
            features = model(frame_batch)

        # Move features back to CPU and convert to NumPy array for saving
        features_np = features.cpu().numpy()

        for feature_vec, frame_label in zip(features_np, labels):
            # Convert to dictionary format for DataFrame
            row = {'label': frame_label}
            for j, val in enumerate(feature_vec):
                row[f'f{j}'] = val
            all_data.append(row)

    # Save all features to a single CSV file on Google Drive
    df = pd.DataFrame(all_data)
    df.to_csv(feature_output_path, index=False)
    log.info(f"Features saved to: {feature_output_path} ({len(df)} total frames)")
