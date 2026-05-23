import pandas as pd
from pathlib import Path
import glob

# Define the path where your folders are located.
# For example, if your structure is: ./data/DeepSeekLarge/file.csv
# This glob pattern finds all CSVs within subdirectories of a target folder.
search_path = "/Users/luisaporzio/Projects/teachme-project/src/data_validation/NEW_final_wide_format_dataset/*/*.csv"

# Find all matching CSV files
file_paths = glob.glob(search_path)

dataframes = []

for file_path in file_paths:
    # Read the dataset exactly as it is
    df = pd.read_csv(file_path)
    
    # Extract the name of the originating folder using pathlib
    folder_name = Path(file_path).parent.name
    
    # Add the new column
    df['model_name'] = folder_name
    
    # Append to our list
    dataframes.append(df)

# Concatenate all datasets into one large DataFrame
if dataframes:
    aggregated_df = pd.concat(dataframes, ignore_index=True)
    
    # Export to a new CSV file
    output_filename = "NEW_aggregated_final_wide_format_dataset.csv"
    aggregated_df.to_csv(output_filename, index=False)
    print(f"Successfully aggregated {len(dataframes)} files into '{output_filename}'.")
else:
    print("No CSV files found in the specified path.")