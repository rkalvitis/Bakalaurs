
import os
import csv
import re
import pandas as pd


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