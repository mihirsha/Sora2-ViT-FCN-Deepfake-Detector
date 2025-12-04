from torchvision import models
from tqdm import tqdm
from utils.load_config import load_config
from utils.logger import get_logger
import torch.nn as nn
import torch
import pandas as pd
from pathlib import Path
from utils.video_processing import VideoFrameExtractor
from sklearn.model_selection import train_test_split

log = get_logger(__name__)

config = load_config()
DEVICE = config['model']['DEVICE']
FEATURE_OUTPUT_FILE = config["training"]["feature_output_file"]
FEATURE_DIM = int(config["model"]["feature_dim"])
PATH_TO_FAKE_VIDEOS = config["training"]["fake_videos_sub_dir"]
PATH_TO_REAL_VIDEOS = config["training"]["real_videos_sub_dir"]

PATH_TO_TRAIN_CSV = config["training"]["train_features_sub_dir"]
PATH_TO_VAL_CSV = config["training"]["val_features_sub_dir"]

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

        # Create a unique identifier for each video
        unique_video_id = f"{label}_{video_path.name}"

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
            row = {'label': frame_label, 'video_id': unique_video_id}
            for j, val in enumerate(feature_vec):
                row[f'f{j}'] = val
            all_data.append(row)

    # Save all features to a single CSV file on Google Drive
    df = pd.DataFrame(all_data)
    df.to_csv(feature_output_path, index=False)
    log.info(f"Features saved to: {feature_output_path} ({len(df)} total frames)")



def create_split_feature_files(test_size=0.3, random_state=42):
    """
    Reads the master feature CSV, splits it by unique Video ID, and saves the
    training and validation features to separate CSV files on disk.
    """

    # Define new output paths for the split files
    TRAIN_CSV_PATH = PATH_TO_TRAIN_CSV
    VAL_CSV_PATH = PATH_TO_VAL_CSV

    print(f" Loading master features from: {FEATURE_OUTPUT_FILE}")
    try:
        features_df = pd.read_csv(FEATURE_OUTPUT_FILE)
    except FileNotFoundError:
        print(" Error: Master feature file not found. Run Phase 1 (Extraction) first.")
        return None, None
    except Exception as e:
        print(f" Error reading master feature file: {e}")
        return None, None

    # 1. Validate Video ID Column
    if 'video_id' not in features_df.columns:
        print(" CRITICAL ERROR: 'video_id' column not found in CSV.")
        print("Please rerun Phase 1 to include the unique video identifier.")
        return None, None

    # 2. Split the unique Video IDs

    unique_videos_df = features_df.groupby('video_id').first().reset_index()

    unique_video_ids = unique_videos_df['video_id'].values
    video_labels = unique_videos_df['label'].values

    # 2. Perform STRATIFIED Split on Video IDs
    # Stratify = video_labels ensures the train and val sets have the same ratio
    # of 0s (real) and 1s (fake) as the original set.

    train_ids, val_ids, _, _ = train_test_split(
        unique_video_ids,
        video_labels,
        test_size=test_size,
        random_state=random_state,
        stratify=video_labels # Ensures balanced class split
    )

    # 3. Filter the DataFrame (Group-Level Split)

    # Select all frames belonging to the training video IDs
    train_df = features_df[features_df['video_id'].isin(train_ids)].copy()

    # Select all frames belonging to the validation video IDs
    val_df = features_df[features_df['video_id'].isin(val_ids)].copy()

    # 4. Drop the temporary 'video_id' column and Save Files

    # Drop video_id column before saving as it is not needed for training
    train_df = train_df.drop(columns=['video_id'])
    val_df = val_df.drop(columns=['video_id'])

    train_df.to_csv(TRAIN_CSV_PATH, index=False)
    val_df.to_csv(VAL_CSV_PATH, index=False)

    print(f"✅ Split Complete: Training set saved ({len(train_ids)} videos, {len(train_df)} frames).")
    print(f"✅ Split Complete: Validation set saved ({len(val_ids)} videos, {len(val_df)} frames).")
    print(f"Saved to: {TRAIN_CSV_PATH} and {VAL_CSV_PATH}")

    return TRAIN_CSV_PATH, VAL_CSV_PATH
