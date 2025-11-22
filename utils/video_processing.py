import cv2
from PIL import Image
from utils.load_config import load_config

config = load_config()
DEVICE = config['model']['DEVICE']
FRAME_RATE_FPS = int(config["video_processing"]["frame_sample_rate_fps"])

class VideoFrameExtractor:
    """Extracts and preprocesses frames from a video file."""
    def __init__(self, preprocess_func, sample_rate_fps=FRAME_RATE_FPS):
        self.preprocess = preprocess_func
        self.sample_rate_fps = sample_rate_fps

    def extract(self, video_path, label):

        print(f"Processing: {video_path}")
        
        # Explicitly convert Path object to string for robust cv2 compatibility
        cap = cv2.VideoCapture(str(video_path))
        
        # Ensure we read the file
        if not cap.isOpened():
            print(f"Error: Failed to open video file: {video_path}")
            return [], []

        fps = cap.get(cv2.CAP_PROP_FPS)
        # Calculate how many native frames to skip to meet the target sample_rate_fps
        if fps > 0:
            frame_skip_interval = int(round(fps / self.sample_rate_fps))
            if frame_skip_interval == 0:
                frame_skip_interval = 1
        else:
            frame_skip_interval = 30 
            
        all_features = []
        all_labels = []
        
        frame_index = 0
        
        while True:
            read_success, frame = cap.read()
            
            if not read_success:
                # Break if the video has ended or reading failed
                break

            # Sample frames based on calculated skip interval
            if frame_index % frame_skip_interval == 0:
                
                # Convert BGR (OpenCV) to RGB 
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # CRITICAL FIX: Convert NumPy array to PIL Image object
                frame_pil = Image.fromarray(frame_rgb)

                # Preprocess: Apply ViT transforms (Resize, Normalize, ToTensor)
                img_tensor = self.preprocess(frame_pil)

                all_features.append(img_tensor)
                all_labels.append(label)

            frame_index += 1

        cap.release()
        return all_features, all_labels