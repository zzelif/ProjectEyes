import os
import subprocess
from datetime import datetime

timestamp = datetime.now().strftime("%Y-%m-%d")

def run_openface(dataset_dir, output_dir, feature_extractor_path):
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                output_path = os.path.join(output_dir, timestamp)
                os.makedirs(output_path, exist_ok=True)
                subprocess.run([feature_extractor_path, '-f', image_path, '-out_dir', output_path, '-aus'])
