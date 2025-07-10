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
    images_path = next((p for p in dataset_paths if 'data_object_image_2.zip' in p), None)
    labels_path = next((p for p in dataset_paths if 'data_object_label_2.zip' in p), None)
    spec_file_path = next((p for p in dataset_paths if 'detectnet_v2_tfrecords_kitti_trainval_updated.txt' in p), None)

    if not all([images_path, labels_path, spec_file_path]):
        raise FileNotFoundError("Missing one or more required input files.")
    
    return images_path, labels_path, spec_file_path

def modify_spec_file(spec_file_path, new_class_mappings=None):
    with open(spec_file_path, 'r') as file:
        spec_content = file.read()

    #Patch the root directory path
    correct_data_path = "/workspace/tao-experiments/data/training"

    spec_content = re.sub(r'root_directory_path:\s*".*?"',
                          f'root_directory_path: "{correct_data_path}"',
                          spec_content)

    spec_content = re.sub(r'image_directory_path:\s*".*?"',
                          f'image_directory_path: "{correct_data_path}"',
                          spec_content)

    spec_content = re.sub(r'image_dir_name:\s*".*?"', 'image_dir_name: "image_2"', spec_content)
    spec_content = re.sub(r'label_dir_name:\s*".*?"', 'label_dir_name: "label_2"', spec_content)

    spec_content = re.sub(r'val_split:\s*\d+(\.\d+)?', 'val_split: 14', spec_content)

    if new_class_mappings:
        for old_class, new_class in new_class_mappings.items():
            spec_content = spec_content.replace(f'key: "{old_class}"', f'key: "{new_class}"')

    tf_temp_dir = tempfile.mkdtemp()
    modified_spec_file_path = os.path.join(tf_temp_dir, 'detectnet_v2_tfrecords_kitti_trainval_updated.txt')
    with open(modified_spec_file_path, 'w') as file:
        file.write(spec_content)    

    return modified_spec_file_path

if __name__ == "__main__":
    # Setup env vars
    os.environ["TAO_DOCKER_DISABLE"] = "1"
    os.environ["TF_ALLOW_IOLIBS"] = "1"
    ngc_key = os.environ.get("NGC_API_KEY")
    if ngc_key is None:
        raise ValueError("NGC_API_KEY environment variable is not set!")
    os.environ["KEY"] = ngc_key
    os.environ["NUM_GPUS"] = "1"
    os.environ["USER_EXPERIMENT_DIR"] = "/valohai/outputs/tao-experiments/detectnet_v2"
    os.environ["DATA_DOWNLOAD_DIR"] = "/workspace/tao-experiments/data"
    os.environ["SPECS_DIR"] = "/workspace/tao-experiments/detectnet_v2/specs"
    os.environ["LOCAL_PROJECT_DIR"] = "/project"
    os.environ["LOCAL_EXPERIMENT_DIR"] = os.path.join(os.environ["LOCAL_PROJECT_DIR"], "detectnet_v2")
    os.environ["TAO_DOCKER_CONFIG_OVERRIDE"] = "1"

    images_path, labels_path, spec_file_path = get_dataset_paths()

    tf_records_archive = valohai.inputs("tfrecords").path(process_archives=False)
    unzip_dir = "/workspace/tao-experiments/data/tfrecords"
    os.makedirs(unzip_dir, exist_ok=True)

    shutil.unpack_archive(tf_records_archive, extract_dir=unzip_dir, format='zip')

    print("Unzipped TFRecords to:", unzip_dir)

    #recirsively list all files and directories
    for root, dirs, files in os.walk(unzip_dir):
        print(f"Directory: {root}")
        for d in dirs:
            print(f"  Subdirectory: {d}")
        for f in files:
            print(f"  File: {f}")


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

    final_training_dir = "/workspace/tao-experiments/data/training"
    shutil.copytree(image_dir, os.path.join(final_training_dir, 'image_2'), dirs_exist_ok=True)
    shutil.copytree(label_dir, os.path.join(final_training_dir, 'label_2'), dirs_exist_ok=True)

    mounts_file = os.path.expanduser("~/.tao_mounts.json")
    drive_map = {
        "Mounts": [
            {"source": os.environ["LOCAL_PROJECT_DIR"], "destination": "/workspace/tao-experiments"},
            {"source": os.path.dirname(spec_file_path), "destination": os.environ["SPECS_DIR"]}
        ]
    }
    with open(mounts_file, "w") as mfile:
        json.dump(drive_map, mfile, indent=4)
    # Dynamically locate the base directory
    # Correct location should point to /training
    kitti_root = os.path.join(temp_dir, 'tao_data', 'training')
    # Modify spec file using the correct path
    modified_spec = modify_spec_file(
        spec_file_path=spec_file_path,
    )

    # Save modified spec to mounted output dir
    specs_target_dir = os.path.join(os.environ["LOCAL_PROJECT_DIR"], "detectnet_v2", "specs")
    os.makedirs(specs_target_dir, exist_ok=True)

    final_spec_path = os.path.join(specs_target_dir, os.path.basename(modified_spec))
    shutil.copy(modified_spec, final_spec_path)

    # Make sure output folder exists
    tfrecord_output_dir = "/workspace/tao-experiments/data/tfrecords/kitti_trainval"
    os.makedirs(tfrecord_output_dir, exist_ok=True)

    # Confirm SPECS_DIR points to correct directory
    os.environ["SPECS_DIR"] = specs_target_dir
    
    eval_cmd = [
        "tao", "model", "detectnet_v2", "evaluate",
        "-e",  list(valohai.inputs("train_specs").paths())[0],
        "-m", list(valohai.inputs("model").paths())[0],
        "-k", os.environ["KEY"],
    ]

    print("🚀 Launching TAO evaluation...")

    eval_process = subprocess.Popen(eval_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    # Stream output line by line
    for line in eval_process.stdout:
        print(line, end='')  # already includes newline

    eval_process.wait()

    if eval_process.returncode != 0:
        print(f"Evaluation failed with return code {eval_process.returncode}")
    else:
        print("Evaluation complete.")