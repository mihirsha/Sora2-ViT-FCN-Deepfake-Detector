import torch
import numpy as np
from pathlib import Path
from utils.load_config import load_config
from utils.logger import get_logger
from utils.video_processing import VideoFrameExtractor
import torch.nn as nn
from model.fcn_classifier import FCN_Classifier
from model.feature_extractor import setup_feature_extractor

log = get_logger(__name__)

config = load_config()
DEVICE = config['model']['DEVICE']
FEATURE_DIM = int(config["model"]["feature_dim"])
MODEL_WEIGHTS_PATH = config['paths']['model_weights_path']
THRESHOLD = config['testing']['threshold']
SAMPLE_RATE = config['testing']['sample_rate_fps']


def predict_video_level(model, video_path, feature_extractor_model, preprocess_fn, sample_rate=10):
    """
    Performs frame-level prediction and temporal aggregation for a single video.
    ... (full function definition from previous response) ...
    """
    extractor = VideoFrameExtractor(preprocess_fn, sample_rate_fps=sample_rate)
    frames_tensors, _ = extractor.extract(video_path, 0)
    
    if not frames_tensors:
        log.warning(f"Could not extract any frames from {video_path}")
        return 0.5, np.array([0.5])

    frame_batch = torch.stack(frames_tensors).to(DEVICE)
    
    with torch.no_grad():
        feature_extractor_model.eval()
        features = feature_extractor_model(frame_batch)
    
    model.eval()
    with torch.no_grad():
        logits = model(features)
        frame_scores = torch.sigmoid(logits).cpu().numpy().flatten()
    
    video_score = np.mean(frame_scores)
    return video_score, frame_scores


def main():
    """Main execution function for video deepfake detection."""
    # Configuration
    TEST_VIDEO_DIR = Path(config['testing']['test_videos_sub_dir'])

    # 1. Load the frozen feature extractor (ViT)
    feature_extractor_model, preprocess_fn = setup_feature_extractor()

    # 2. Load the trained FCN classifier weights
    trained_fcn = FCN_Classifier(FEATURE_DIM)
    trained_fcn.load_state_dict(torch.load(MODEL_WEIGHTS_PATH))
    trained_fcn.to(DEVICE).eval()
    
    # Get all test videos
    test_video_paths = list(TEST_VIDEO_DIR.glob("*.mp4"))
    
    if not test_video_paths:
        log.warning(f"No videos found in {TEST_VIDEO_DIR}")
        return
    
    log.info(f"Found {len(test_video_paths)} videos to process")
    
    # Process each video
    for video_path in test_video_paths:
        # Run prediction
        video_probability, frame_scores = predict_video_level(
            trained_fcn,
            video_path,
            feature_extractor_model,
            preprocess_fn,
            sample_rate=SAMPLE_RATE
        )
        
        # Determine prediction label
        prediction_label = "FAKE (AI-Generated)" if video_probability > THRESHOLD else "REAL (Authentic)"
        
        # Log results
        log.info("--- TEST RESULTS ---")
        log.info(f"Video Path: {video_path}")
        log.info(f"Video Score (Probability of FAKE): {video_probability:.4f}")
        log.info(f"Final Detection: **{prediction_label}**")
        log.info(f"Total Frames Analyzed: {len(frame_scores)}")
        log.info(f"Frame Score Range: {np.min(frame_scores):.2f} to {np.max(frame_scores):.2f}")

# Execute main function
if __name__ == "__main__":
    main()