from valohai import inputs

def get_dataset_paths():
    dataset_paths = list(inputs("dataset").paths(process_archives=False))
    images_path = next((p for p in dataset_paths if "images.zip" in p), None)
    labels_path = next((p for p in dataset_paths if "labels.zip" in p), None)

    if not all([images_path, labels_path]):
        raise FileNotFoundError("Missing one or more required input files.")

    return images_path, labels_path