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
import tensorflow as tf
import valohai

def get_dataset_paths():
    dataset_paths = list(inputs('dataset').paths(process_archives=False))
    images_path = next((p for p in dataset_paths if 'data_object_image_2.zip' in p), None)
    labels_path = next((p for p in dataset_paths if 'data_object_label_2.zip' in p), None)
    spec_file_path = next((p for p in dataset_paths if 'detectnet_v2_tfrecords_kitti_trainval_updated.txt' in p), None)
    

    if not all([images_path, labels_path, spec_file_path]):
        raise FileNotFoundError("Missing one or more required input files.")

    return images_path, labels_path, spec_file_path

# Auto-detect subdir containing 'training/image_2'
def find_kitti_root(base_path):
    for root, dirs, files in os.walk(base_path):
        if 'image_2' in dirs and os.path.basename(root) == 'training':
            print(f"✅ Found KITTI training dir at: {root}")
            # Go up one level to get the path that contains /training
            return os.path.dirname(root)
    raise FileNotFoundError("training/image_2 not found in extracted dataset.")

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

    print(f"✅ Spec file updated and saved at {modified_spec_file_path}")
    return modified_spec_file_path

def modify_training_spec_file(original_spec_path, tfrecords_path, image_dir):
    with open(original_spec_path, 'r') as file:
        content = file.read()

    # Build new data_sources block
    new_data_sources = f"""
data_sources {{
  tfrecords_path: "{tfrecords_path}"
  image_directory_path: "{image_dir}"
}}
"""

    # Replace existing data_sources or append if missing
    if "data_sources" in content:
        content = re.sub(r"data_sources\s*{[^}]*}", new_data_sources, content, flags=re.DOTALL)
    else:
        content += "\n" + new_data_sources

    content = re.sub(
    r'^\s*pretrained_model_file:\s*".*?"\s*$',
    f'pretrained_model_file: "{valohai.inputs("pretrained_model").path(process_archives=False)}"',
    content,
    flags=re.MULTILINE
)

    content = re.sub(r'num_epochs:\s*\d+', 'num_epochs: 5', content)

    # Save modified spec to a temp location
    modified_spec_path = os.path.join(tempfile.mkdtemp(), "detectnet_v2_train_modified.txt")
    with open(modified_spec_path, "w") as file:
        file.write(content)

    print(f"Training spec patched at: {modified_spec_path}")
    with open(modified_spec_path, "r") as f:
        print("Patched training spec content:\n", f.read())

    # Save to Valohai outputs
    valohai.outputs("my-output").path(os.path.basename(modified_spec_path))

    return modified_spec_path



if __name__ == "__main__":
    # Setup env vars
    os.environ["TAO_DOCKER_DISABLE"] = "1"
    os.environ["TF_ALLOW_IOLIBS"] = "1"
    ngc_key = os.environ.get("NGC_API_KEY")
    if ngc_key is None:
        raise ValueError("NGC_API_KEY environment variable is not set!")
    os.environ["KEY"] = ngc_key
    os.environ["TAO_DOCKER_CONFIG_OVERRIDE"] = "1"

    images_path, labels_path, spec_file_path = get_dataset_paths()
    print("Images path:", images_path)
    print("Labels path:", labels_path)
    print("Spec path:", spec_file_path)

    # Unpack into correct folder structure under DATA_DOWNLOAD_DIR
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

    # ✅ Now check them
    print("📂 Contents of image_2:", os.listdir(image_dir)[:5])
    print("📂 Contents of label_2:", os.listdir(label_dir)[:5])     

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

    # Launch TFRecord conversion
    convert_cmd = [
        "tao", "model", "detectnet_v2", "dataset_convert",
        "-d", final_spec_path,
        "-o", tfrecord_output_dir
    ]


    # Ensure the output directory exists

    print("📄 Final spec file before conversion:")
    with open(final_spec_path, 'r') as f:
        print(f.read())


    print("🚀 Launching TAO dataset_convert...")
    result = subprocess.run(convert_cmd, text=True, capture_output=True)
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    if result.returncode != 0:
        print("❌ converting failed.")
    else:
        print("✅ converting complete.")



    source_dir = "/workspace/tao-experiments/data/tfrecords"
    zip_output_path = valohai.outputs("my-output").path("all_tfrecords.zip")

    shutil.make_archive(zip_output_path.replace(".zip", ""), 'zip', source_dir)

    print(f"TFRecords Zipped and saved to: {zip_output_path}")


        # Get the file path from the input
    original_train_spec = list(valohai.inputs("train_specs").paths())[0]
    train_spec_path = modify_training_spec_file(
        original_spec_path=original_train_spec,
        tfrecords_path="/workspace/tao-experiments/data/tfrecords/kitti_trainval-fold-*",
        image_dir="/workspace/tao-experiments/data/training"
    )



#######################################################

    train_cmd = [
        "tao", "model", "detectnet_v2", "train",
        "-e", train_spec_path,
        "-r", os.environ["USER_EXPERIMENT_DIR"],
        "-k", os.environ["KEY"],
        "--gpus", os.environ["NUM_GPUS"],
    ]

    print("🚀 Launching TAO training...")

    # Regex pattern to match the key values from TAO log
    pattern = re.compile(
        r"epoch = ([\d\.]+), learning_rate = ([\de\+\-\.]+), loss = ([\d\.e\-]+), step = (\d+)"
    )

    # Prepare metadata output file
    metadata_path = valohai.outputs().path("valohai.metadata.jsonl")
    with open(metadata_path, "w") as outfile:
        process = subprocess.Popen(train_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

        for line in process.stdout:
            print(line, end="")  # Still show it in console

            # Try to extract metrics
            match = pattern.search(line)
            if match:
                epoch, lr, loss, step = match.groups()
                metadata = {
                    "epoch": float(epoch),
                    "learning_rate": float(lr),
                    "loss": float(loss),
                    "step": int(step)
                }
                print(json.dumps(metadata))

        process.wait()

    if process.returncode != 0:
        print(f"❌ Training failed with return code {process.returncode}")
    else:
        print("✅ Training complete.")


