import os
import random
import shutil
from math import ceil
from collections import defaultdict
import csv
import re
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
import pandas as pd
import torchvision.transforms as transforms

#function selects data from directory based on precentage passde to function, images are taken randomly 
def SelectDataByPercent(src_dir, dest_dir, percentage):
    # Create a mapping of subcategories to image paths
    data_map = defaultdict(lambda: {'good': [], 'bad': []})
    
    # Traverse the directory hierarchy
    for root, dirs, files in os.walk(src_dir):
        if files:
            rel_path = os.path.relpath(root, src_dir)
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')):
                    category = 'good' if 'good' in rel_path else 'bad'
                    data_map[rel_path][category].append(os.path.join(root, file))
    
    # Prepare the destination directory if it doesn't exist
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    # Process each subcategory to copy the specific percentage
    for rel_path, subcategories in data_map.items():
        good_images = subcategories['good']
        bad_images = subcategories['bad']
        
        # Calculate the number of images to copy per category based on percentage
        num_good_to_copy = ceil(len(good_images) * (percentage / 100))
        num_bad_to_copy = ceil(len(bad_images) * (percentage / 100))

        #TODO
        #random.seed(10)
        # Randomly sample the specified percentage of images from each category
        chosen_good_images = random.sample(good_images, num_good_to_copy) if num_good_to_copy > 0 else []
        chosen_bad_images = random.sample(bad_images, num_bad_to_copy) if num_bad_to_copy > 0 else []
        
        chosen_images = chosen_good_images + chosen_bad_images
        
        # Ensure the destination path exists
        dest_path = os.path.join(dest_dir, rel_path)
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
        
        # Copy the chosen images to the destination
        for img_path in chosen_images:
            shutil.copy(img_path, dest_path)


def create_synthetic_csv_from_real_and_structure(real_csv_path, synthetic_root, synthetic_csv_filename):
    # Read the real image CSV file into a dictionary keyed by imageID (excluding ".png" extension)
    real_data = {}
    with open(real_csv_path, mode='r', newline='') as real_csv_file:
        # Use csv.reader to print the headers for debugging purposes
        reader = csv.reader(real_csv_file)
        headers = next(reader)
        headers = [header.strip() for header in headers]  # Strip spaces from headers
        
        print(f"Headers in the CSV: {headers}")  # Debugging step
        
        if 'imageID' not in headers:
            raise ValueError("CSV file does not have an 'imageID' column.")
        
        # Rewind the file to read the whole data with DictReader
        real_csv_file.seek(0)
        reader = csv.DictReader(real_csv_file)
        for row in reader:
            image_id = row['imageID'].replace(".png", "")
            real_data[image_id] = row
    
    # Regular expression to parse the synthetic filenames
    pattern = re.compile(r"synthetic_\d+_(\d+_\d+).png")

    # Initialize the synthetic data list
    synthetic_data = []
    
    # Traverse the synthetic image folder structure
    for decision in ['good', 'bad']:
        decision_folder = os.path.join(synthetic_root, decision)
        for cell_type_folder in os.listdir(decision_folder):
            cell_type_path = os.path.join(decision_folder, cell_type_folder)
            if os.path.isdir(cell_type_path):
                for day_folder in os.listdir(cell_type_path):
                    day_path = os.path.join(cell_type_path, day_folder)
                    if os.path.isdir(day_path):
                        for filename in os.listdir(day_path):
                            match = pattern.match(filename)
                            if match:
                                real_image_id = match.group(1)
                                if real_image_id in real_data:
                                    row = real_data[real_image_id]
                                    synthetic_image_id = filename.replace(".png", "")
                                    synthetic_data.append({
                                        "imageID": synthetic_image_id,
                                        "cell type": row["cell type"],
                                        "seeding density, cells/ml": row["seeding density, cells/ml"],
                                        "time after seeding, h": row["time after seeding, h"],
                                        "day": row["day"],
                                        "Decision 1/2 (good/bad)": row["Decision 1/2 (good/bad)"],
                                        "flow rate": row["flow rate"]
                                    })

    # Write the collected synthetic data to a new CSV file
    with open(synthetic_csv_filename, mode='w', newline='') as synthetic_csv_file:
        fieldnames = ["imageID", "cell type", "seeding density, cells/ml", "time after seeding, h", 
                      "day", "Decision 1/2 (good/bad)", "flow rate"]
        writer = csv.DictWriter(synthetic_csv_file, fieldnames=fieldnames, delimiter=',')
        writer.writeheader()
        writer.writerows(synthetic_data)


def GetTransformer(height, width):
    transform = transforms.Compose([
        transforms.Resize((height, width)),  # EfficientNet B7 uses a larger input size
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform



class CellDataset(Dataset):
    def __init__(self, csv_file, base_dir, transform=None):
        """
        Args:
            csv_file (str): Path to the CSV file containing image metadata.
            base_dir (str): Base directory containing the "train", "val", and "test" folders.
            transform (callable, optional): A function/transform to apply to the images.
        """
        # Read the CSV file into a DataFrame
        self.data = pd.read_csv(csv_file)
        self.base_dir = base_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Extract relevant information from the CSV file
        img_id = self.data.iloc[idx, 0]
        split = self.data.iloc[idx, 1]  # e.g., train, val, test
        label = self.data.iloc[idx, -2]  # good or bad
        cell_type = self.data.iloc[idx, 2]  # e.g., 'cell_type_A549'
        days = self.data.iloc[idx, 3]  # e.g., '0-1_days'

        # Construct the relative image path based on the folder hierarchy
        img_path = os.path.join(self.base_dir, split, label, cell_type, days, f"{img_id}.png")

        try:
            # Attempt to load the image and convert it to RGB
            image = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, UnidentifiedImageError) as e:
            print(f"Error loading image {img_path}: {e}")
            return None, None

        # Transform the label to binary if "good" or "bad"
        label = 1 if label == "good" else 0

        # Apply transformations if any
        if self.transform:
            image = self.transform(image)

        return image, label
