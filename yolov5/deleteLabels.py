import os

# Define the paths to the directories containing the image and label files
image_directory = 'C:/Users/dammi/Documents/OcxYolo5/yolov5/dataset/train/images'  # Update this to your images directory path
label_directory = 'C:/Users/dammi/Documents/OcxYolo5/yolov5/dataset/train/labels'  # Update this to your labels directory path

# Get a list of all image and label files from their respective directories
image_files = [f for f in os.listdir(image_directory) if os.path.isfile(os.path.join(image_directory, f))]
label_files = [f for f in os.listdir(label_directory) if os.path.isfile(os.path.join(label_directory, f))]

# Extract the base names without extension for comparison
image_base_names = [os.path.splitext(image)[0] for image in image_files]
label_base_names = [os.path.splitext(label)[0] for label in label_files]

# Find label files that do not have a matching image file
labels_to_delete = [label for label in label_base_names if label not in image_base_names]

# Now, construct the full paths to the files to be deleted
full_paths_to_delete = [os.path.join(label_directory, label + '.txt') for label in labels_to_delete]

# Initialize progress counters
total_files = len(full_paths_to_delete)
deleted_count = 0

print(f"Starting deletion of {total_files} label files...")

# Delete the files
for file_path in full_paths_to_delete:
    if os.path.exists(file_path):
        print(f"Deleting {file_path}...")
        os.remove(file_path)
        deleted_count += 1
    else:
        print(f"Label file {file_path} does not exist and cannot be deleted.")

# Final report
print(f"Deleted {deleted_count} out of {total_files} label files.")
