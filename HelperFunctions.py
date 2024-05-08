
import os
import csv
import pandas as pd
from glob import glob
import shutil


def copy_directory(src_dir: str, dest_dir: str) -> None:
    """
    Copy a directory and all its contents to a specific location.

    :param src_dir: Path of the source directory to copy.
    :param dest_dir: Destination directory path where the source directory should be copied.
    """
    # Normalize and convert paths to absolute paths
    src_dir = os.path.abspath(src_dir)
    dest_dir = os.path.abspath(dest_dir)

    # Check if the source directory exists and is a directory
    if not os.path.isdir(src_dir):
        raise ValueError(f"Source path '{src_dir}' is not a directory or does not exist.")
    
    # Ensure the destination directory exists
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Copy the source directory into the destination directory
    shutil.copytree(src_dir, os.path.join(dest_dir, os.path.basename(src_dir)), dirs_exist_ok=True)

    print(f"Copied '{src_dir}' to '{os.path.join(dest_dir, os.path.basename(src_dir))}' successfully.")

def delete_file(file_path):
    """Delete a specific file if it exists."""
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
    else:
        print(f"The file {file_path} does not exist.")

#joins two csv files together
def concatenate_csv(file1, file2, output_file):
    with open(file1, mode='r', newline='') as f1, open(file2, mode='r', newline='') as f2, open(output_file, mode='w', newline='') as output:
        reader1 = csv.reader(f1)
        reader2 = csv.reader(f2)
        writer = csv.writer(output)

        # Read the header from the first file
        headers = next(reader1)
        # Write the header to the output file
        writer.writerow(headers)

        # Write all rows from the first file
        for row in reader1:
            writer.writerow(row)

        # Skip the header in the second file
        next(reader2)

        # Write all rows from the second file
        for row in reader2:
            writer.writerow(row)


#coverts xlsx to csv
def convert_xlsx_to_csv(xlsx_file, csv_file):
    # Load the Excel file
    excel_data = pd.read_excel(xlsx_file, sheet_name=None)
    
    # If there's more than one sheet, prompt the user for which one to use
    if len(excel_data) > 1:
        print(f"Available sheets: {', '.join(excel_data.keys())}")
        sheet_name = input("Please specify the sheet to convert: ")
    else:
        sheet_name = list(excel_data.keys())[0]  # Take the first and only sheet

    # Read the selected sheet into a DataFrame
    df = excel_data[sheet_name]
    
    # Write the DataFrame to a CSV file
    df.to_csv(csv_file, index=False)


def collect_image_paths(main_directory):
    """
    Collects all image paths from the main directory following the specific structure.
    
    Args:
    - main_directory (str): Path to the main dataset directory.

    Returns:
    - image_paths (dict): A mapping of image names to their full paths and dataset type.
    """
    image_paths = {}

    for dataset_type in ['train', 'val', 'test']:
        dataset_path = os.path.join(main_directory, dataset_type)
        # Recursively search all subdirectories for image files
        for image_file in glob(f'{dataset_path}/**/*.png', recursive=True):
            image_name = os.path.basename(image_file).lower()
            image_paths[image_name] = (image_file, dataset_type)

    print(f"Collected {len(image_paths)} image paths.")
    return image_paths

def split_datasets_by_actual_images(csv_path, main_directory, output_directory):
    """
    Splits a CSV file containing image training data into train, val, and test datasets,
    ensuring that only images present in the specified directory are considered.
    
    Args:
    - csv_path (str): Path to the original CSV file.
    - main_directory (str): Path to the main directory containing the train, val, and test folders.
    - output_directory (str): Path to the directory where the split CSV files will be stored.
    """
    # Load the CSV file into a DataFrame
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        return

    # Collect all actual image paths and their dataset types
    image_paths = collect_image_paths(main_directory)

    # Function to determine the dataset type by checking against collected paths
    def get_dataset_type(image_id):
        image_name_with_extension = f"{image_id.lower()}.png"
        return image_paths.get(image_name_with_extension, (None, None))[1]

    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Identify dataset type for each imageID and filter out rows where no image file is found
    df['dataset_type'] = df['imageID'].apply(get_dataset_type)
    df_filtered = df.dropna(subset=['dataset_type'])

    print(f"Filtered dataset contains {len(df_filtered)} records.")

    # Split and save the datasets based on the type without adding the 'dataset_type' column
    for dataset_type in ['train', 'val', 'test']:
        finalOutputDir = os.path.join(output_directory, dataset_type)
        dataset_df = df_filtered[df_filtered['dataset_type'] == dataset_type].drop(columns=['dataset_type'])
        if not dataset_df.empty:
            output_path = os.path.join(finalOutputDir, f'{dataset_type}_dataset.csv')
            dataset_df.to_csv(output_path, index=False)
            print(f"{dataset_type.capitalize()} data saved to {output_path}")
        else:
            print(f"No data available for {dataset_type}.")