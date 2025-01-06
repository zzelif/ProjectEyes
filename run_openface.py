import os
import subprocess
from action_units import aggregate_au_realtime

def run_openface(dataset_dir, output_dir, feature_extractor_path, timestamp):
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                output_path = os.path.join(output_dir, timestamp)
                os.makedirs(output_path, exist_ok=True)
                subprocess.run([feature_extractor_path, '-f', image_path, '-out_dir', output_path, '-aus'])


def _realtime_openface(dir_path, output_dir, extractor_path):
    process = subprocess.Popen(
        [extractor_path, '-fdir', dir_path, '-aus', '-out_dir', output_dir],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    process.communicate()

    print(f"Succesfully extracted the action units into {output_dir}")