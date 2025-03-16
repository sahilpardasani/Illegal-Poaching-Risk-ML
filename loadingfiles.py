import os
import psycopg2

# Database connection parameters
db_config = {
    'dbname': 'wildlifedata',
    'user': 'postgres',
    'password': 'ot2266',
    'host': 'localhost',
    'port': '5432'
    
}

# Directory containing CSV files
csv_directory = '/Users/sahil.pardasani/Desktop/Illegal Poaching Project/Trade_database_download_v2024.1'

# Table name in PostgreSQL
table_name = 'trades_data'

# Connect to PostgreSQL
conn = psycopg2.connect(**db_config)
cursor = conn.cursor()

# Loop through all CSV files in the directory
for filename in os.listdir(csv_directory):
    if filename.endswith(".csv"):
        csv_file = os.path.join(csv_directory, filename)
        print(f"Processing {csv_file}...")
        try:
            # Use COPY command to load CSV data
            with open(csv_file, 'r') as f:
                cursor.copy_expert(f"COPY {table_name} FROM STDIN WITH CSV HEADER", f)
            conn.commit()
            print(f"Inserted data from {csv_file}")
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            conn.rollback()  # Rollback in case of error

# Close the database connection
cursor.close()
conn.close()
print("All CSV files have been processed.")