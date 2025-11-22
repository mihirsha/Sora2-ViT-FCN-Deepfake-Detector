from model.feature_extractor import setup_feature_extractor
from model.feature_extractor import extract_and_save_features
from utils.load_config import load_config

cfg = load_config()
FEATURE_OUTPUT_FILE = cfg["paths"]["feature_output_file"]

feature_extractor_model, preprocess_fn = setup_feature_extractor()
extract_and_save_features(feature_extractor_model, preprocess_fn, FEATURE_OUTPUT_FILE)
