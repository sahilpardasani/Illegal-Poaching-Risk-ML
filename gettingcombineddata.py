import os
import pandas as pd

# Directory containing CSV files
csv_directory = '/Users/sahil.pardasani/Desktop/Illegal Poaching Project/Trade_database_download_v2024.1'

# Check if the directory exists
if not os.path.exists(csv_directory):
    raise FileNotFoundError(f"Directory not found: {csv_directory}")

# List all CSV files in the directory
csv_files = [f for f in os.listdir(csv_directory) if f.endswith('.csv')]
print(f"Found {len(csv_files)} CSV files in {csv_directory}")

if not csv_files:
    raise ValueError("No CSV files found in the directory.")

# List to store DataFrames
chunks = []

# Loop through CSV files and read them into DataFrames
for filename in csv_files:
    csv_file = os.path.join(csv_directory, filename)
    try:
        # Read the CSV file with low_memory=False to avoid mixed-type warnings
        df = pd.read_csv(csv_file, low_memory=False)
        print(f"File: {filename}, Rows: {len(df)}, Columns: {len(df.columns)}")
        chunks.append(df)  # Append DataFrame to the list
    except Exception as e:
        print(f"Error reading {filename}: {e}")

# Check if any DataFrames were loaded
if not chunks:
    raise ValueError("No valid DataFrames to concatenate.")

# Concatenate all DataFrames
combined_df = pd.concat(chunks, ignore_index=True)
print(f"Combined DataFrame: {len(combined_df)} rows, {len(combined_df.columns)} columns")

# Inspect the combined DataFrame
print(combined_df.head())
print(combined_df.info())

# Save the combined DataFrame to a single CSV file (optional)
output_file = os.path.join(csv_directory, 'combined_data.csv')
combined_df.to_csv(output_file, index=False)
print(f"Combined data saved to {output_file}")