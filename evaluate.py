import os
import tempfile
import shutil
import subprocess
from valohai import inputs
import argparse
import json
import urllib.request
import glob
import zipfile
import re
import valohai



def get_dataset_paths():
    dataset_paths = list(inputs('dataset').paths(process_archives=False))
    images_path = next((p for p in dataset_paths if 'images.zip' in p), None)
    labels_path = next((p for p in dataset_paths if 'labels.zip' in p), None)
    
    if not all([images_path, labels_path]):
        raise FileNotFoundError("Missing one or more required input files.")

    return images_path, labels_path


if __name__ == "__main__":
    # Setup env vars
    ngc_key = os.environ.get("NGC_API_KEY")
    if ngc_key is None:
        raise ValueError("NGC_API_KEY environment variable is not set!")
    os.environ["KEY"] = ngc_key

    images_path, labels_path = get_dataset_paths()

    tf_records_archive = valohai.inputs("tfrecords").path(process_archives=False)
    unzip_dir = os.environ.get("TF_RECORDS_DIR","/workspace/tao-experiments/data/tfrecords/")
    os.makedirs(unzip_dir, exist_ok=True)

    shutil.unpack_archive(tf_records_archive, extract_dir=unzip_dir, format='zip')
    print("Contents of unzip_dir after extraction:", os.listdir(unzip_dir))


    print("Unzipped TFRecords to:", unzip_dir)
    tfrecord_files = glob.glob("/workspace/tao-experiments/data/tfrecords/**/*", recursive=True)
    for file in sorted(tfrecord_files):
        print(file) 

    # Unpack into correct folder structure 
    temp_dir = tempfile.mkdtemp()
    training_dir = os.path.join(temp_dir, 'training')
    # Extract both ZIPs
    with zipfile.ZipFile(images_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    with zipfile.ZipFile(labels_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)

   # Use the actual extracted paths
    image_dir = os.path.join(temp_dir, 'training', 'image_2')
    label_dir = os.path.join(temp_dir, 'training', 'label_2')

    final_training_dir = os.environ.get("TRAINING_DIR","/workspace/tao-experiments/data/training")
    shutil.copytree(image_dir, os.path.join(final_training_dir, 'image_2'), dirs_exist_ok=True)
    shutil.copytree(label_dir, os.path.join(final_training_dir, 'label_2'), dirs_exist_ok=True)

    mounts_file = os.path.expanduser("~/.tao_mounts.json")
    drive_map = {
        "Mounts": [
            {"source": os.environ["LOCAL_PROJECT_DIR"], "destination": "/workspace/tao-experiments"},
        ]
    }
    with open(mounts_file, "w") as mfile:
        json.dump(drive_map, mfile, indent=4)
    # Dynamically locate the base directory
    
    eval_cmd = [
        "tao", "model", "detectnet_v2", "evaluate",
        "-e",  list(valohai.inputs("train_specs").paths())[0],
        "-m", list(valohai.inputs("model").paths())[0],
        "-k", os.environ["KEY"],
    ]

    print("Launching TAO evaluation...")

    eval_process = subprocess.Popen(eval_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    # Stream output line by line
    for line in eval_process.stdout:
        print(line, end='')  # already includes newline

    eval_process.wait()

    if eval_process.returncode != 0:
        print(f"Evaluation failed with return code {eval_process.returncode}")
    else:
        print("Evaluation complete.")