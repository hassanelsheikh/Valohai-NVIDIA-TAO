import argparse
import valohai
import os
import tempfile
import shutil
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from random import sample
from tqdm import tqdm


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Load and visualize KITTI dataset.")
    parser.add_argument('--num_plot_images', type=int, default=10, help='Number of sample images to visualize.')
    parser.add_argument('--subset', action='store_true', help='Enable subset generation')
    parser.add_argument('--subset_num_images', type=int, default=100, help='Number of images for subset')

    return parser.parse_args()



def parse_kitti_labels(label_file_path):
    """Parse a KITTI label file and extract bounding boxes."""
    bboxes = []
    if not os.path.exists(label_file_path):
        return bboxes

    with open(label_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 15:
                label = parts[0]
                xmin = float(parts[4])
                ymin = float(parts[5])
                xmax = float(parts[6])
                ymax = float(parts[7])
                bboxes.append((label, xmin, ymin, xmax, ymax))
    return bboxes

def visualize_kitti_images_with_boxes(image_dir, label_dir, output_dir, num_samples=10):

    all_images = sorted(os.listdir(image_dir))
    sample_images = sample(all_images, min(len(all_images), num_samples))

    for i, img_file in enumerate(sample_images):
        img_path = os.path.join(image_dir, img_file)
        label_file = os.path.join(label_dir, img_file.replace('.png', '.txt'))

        # Load image
        img = plt.imread(img_path)

        # Get image dimensions in pixels
        height, width = img.shape[:2]

        # Define DPI (adjust as needed; 100 is common)
        dpi = 100

        # Compute figure size in inches
        figsize = (width / dpi, height / dpi)

        # Create figure with true pixel size
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.imshow(img)

        # Draw bounding boxes
        bboxes = parse_kitti_labels(label_file)
        for label, xmin, ymin, xmax, ymax in bboxes:
            width = xmax - xmin
            height = ymax - ymin
            rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2,
                                     edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(xmin, ymin - 5, label, color='yellow', fontsize=8, backgroundcolor='black')

        ax.set_title(f"Image: {img_file}")
        ax.axis('off')

        # Save to Valohai output
        save_path = output_dir.path(f"annotated_image_{i + 1}.png")
        plt.savefig(save_path)
        plt.close()
        valohai.outputs().live_upload(save_path)

def generate_kitti_subset(source_image_dir, source_label_dir, output_dir, num_images=100):
    """Create a subset of the KITTI dataset for training."""
    image_out_dir = os.path.join(output_dir, "training", "image_2")
    label_out_dir = os.path.join(output_dir, "training", "label_2")
    os.makedirs(image_out_dir, exist_ok=True)
    os.makedirs(label_out_dir, exist_ok=True)

    all_images = sorted(os.listdir(source_image_dir))
    selected_ids = [f.replace('.png', '') for f in all_images[:num_images]]

    for id in tqdm(selected_ids, desc="Creating KITTI subset"):
        shutil.copy(os.path.join(source_image_dir, f"{id}.png"), os.path.join(image_out_dir, f"{id}.png"))
        shutil.copy(os.path.join(source_label_dir, f"{id}.txt"), os.path.join(label_out_dir, f"{id}.txt"))

    print(f"✅ Created subset with {len(selected_ids)} images at: {output_dir}")


def modify_spec_file(spec_file_path, new_image_dir, new_label_dir, new_class_mappings=None):
    """Modify the spec file with new paths and class mappings."""
    
    # Read the spec file
    with open(spec_file_path, 'r') as file:
        spec_content = file.read()

    # Modify the image and label directories (replace paths with Valohai input paths)
    spec_content = spec_content.replace("/workspace/tao-experiments/data/", new_image_dir)
    spec_content = spec_content.replace("training/image_2", os.path.join(new_image_dir, 'training', 'image_2'))
    spec_content = spec_content.replace("training/label_2", os.path.join(new_label_dir, 'training', 'label_2'))

    # Modify class mappings if provided
    if new_class_mappings:
        for old_class, new_class in new_class_mappings.items():
            spec_content = spec_content.replace(f'key: "{old_class}"', f'key: "{new_class}"')

    # Save the modified spec file back to Valohai outputs
    modified_spec_file_path = valohai.outputs().path('specs/detectnet_v2_tfrecords_kitti_trainval_updated.txt')
    with open(modified_spec_file_path, 'w') as file:
        file.write(spec_content)
    
    print(f"Spec file updated and saved at {modified_spec_file_path}")
    return modified_spec_file_path

if __name__ == "__main__":
    args = parse_args()

    train_archive = valohai.inputs('images').path(process_archives=False)
    labels_archive = valohai.inputs('labels').path(process_archives=False)
    spec_file_path = valohai.inputs('specs').path(process_archives=False)

    if not os.path.exists(train_archive):
        raise FileNotFoundError("Image archive not found.")
    if not os.path.exists(labels_archive):
        raise FileNotFoundError("Label archive not found.")

    with tempfile.TemporaryDirectory() as temp_dir:
        image_dir = os.path.join(temp_dir, 'images')
        label_dir = os.path.join(temp_dir, 'labels')
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)

        if train_archive.endswith('.zip'):
            shutil.unpack_archive(train_archive, image_dir, 'zip')

        if labels_archive.endswith('.zip'):
            shutil.unpack_archive(labels_archive, label_dir, 'zip')

        # Paths to relevant folders
        train_image_path = os.path.join(image_dir, 'training', 'image_2')
        test_image_path = os.path.join(image_dir, 'testing', 'image_2')
        label_path = os.path.join(label_dir, 'training', 'label_2')   

        num_training_images = len(os.listdir(os.path.join(image_dir, 'training', 'image_2')))
        num_training_labels = len(os.listdir(os.path.join(label_dir, 'training', 'label_2')))
        num_testing_images = len(os.listdir(os.path.join(image_dir, 'testing', 'image_2')))

        print("Number of images in the train/val set. {}".format(num_training_images))
        print("Number of labels in the train/val set. {}".format(num_training_labels))
        print("Number of images in the test set. {}".format(num_testing_images))

        #Sample kitti label
        # Display a sample label file
        sample_label_file = os.path.join(label_path, '000110.txt')
        if os.path.exists(sample_label_file):
            print(f"\nContents of sample label file ({sample_label_file}):\n")
            with open(sample_label_file, 'r') as f:
                print(f.read())
        else:
            print(f"Sample label file not found: {sample_label_file}")    

        # Plot a 5 sample images from the training set
        visualize_kitti_images_with_boxes(train_image_path, label_path, valohai.outputs("my_output"), args.num_plot_images)

        new_image_dir = image_dir  # Valohai input for images
        new_label_dir = label_dir  # Valohai input for labels

        modified_spec_file_path = modify_spec_file(spec_file_path, new_image_dir, new_label_dir)
        
        # Save the train and label directories as Valohai outputs

        train_output_dir = valohai.outputs().path('images')
        labels_output_dir = valohai.outputs().path('labels')

        # Copy the ZIP files into the output directory
        shutil.copy(train_archive, os.path.join(train_output_dir, 'data_object_image_2.zip'))
        shutil.copy(labels_archive, os.path.join(labels_output_dir, 'data_object_label_2.zip'))

        metadata = {
            "images/data_object_image_2.zip": {
                "valohai.dataset-versions": [
                    "dataset://KITTI/version2"
                ]
            },
            "labels/data_object_label_2.zip": {
                "valohai.dataset-versions": [
                    "dataset://KITTI/version2"
                ]
            },
            "specs/detectnet_v2_tfrecords_kitti_trainval_updated.txt": {
                "valohai.dataset-versions": [
                    "dataset://KITTI/version2"
                ]
            }
        }

        metadata_path = valohai.outputs().path("valohai.metadata.jsonl")
        with open(metadata_path, "w") as outfile:
            for file_name, file_metadata in metadata.items():
                json.dump({"file": file_name, "metadata": file_metadata}, outfile)
                outfile.write("\n")    


                        
