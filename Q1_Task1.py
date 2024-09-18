import os
import pandas as pd

# Path to the folder where your CSV files are located (same folder as your Python files)
folder_path = "."

# Initialize a list to store the text from all CSV files
all_texts = []

# List of CSV file names and their corresponding text-containing columns
csv_files_info = {
    "CSV1.csv": "SHORT-TEXT",  # For CSV1, the text column is 'SHORT-TEXT'
    "CSV2.csv": "TEXT",        # For CSV2, the text column is 'TEXT'
    "CSV3.csv": "TEXT",        # For CSV3, the text column is 'TEXT'
    "CSV4.csv": "TEXT"         # For CSV4, the text column is 'TEXT'
}

# Iterate through each CSV file and extract the text from the correct column
for file_name, text_column in csv_files_info.items():
    file_path = os.path.join(folder_path, file_name)
    
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Check if the expected text column is in the CSV file
        if text_column in df.columns:
            # Extract the text column and drop any empty values
            texts = df[text_column].dropna().tolist()
            
            # Add extracted texts to the combined list
            all_texts.extend(texts)
        else:
            print(f"Error: Column '{text_column}' not found in {file_name}")
    
    except FileNotFoundError:
        print(f"Error: {file_name} not found in the folder '{folder_path}'")
    except pd.errors.EmptyDataError:
        print(f"Error: {file_name} is empty or corrupted")

# Write the extracted text from all CSV files into a single text file
output_file = os.path.join(folder_path, "combined_texts.txt")
with open(output_file, "w", encoding="utf-8") as f:
    # Join all texts with a newline and write to the .txt file
    f.write("\n".join(all_texts))

print(f"Success: Texts from all CSV files have been extracted and stored in {output_file}")
