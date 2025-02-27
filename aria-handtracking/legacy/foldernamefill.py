import pandas as pd
import os

# Define Directory and CSV Path
image_directory = '/home/totargaming/workspace/sam2/aria-handtracking/HOIST/gathered/'
csv_path = '/home/totargaming/workspace/sam2/aria-handtracking/coordinates.csv'

# Read CSV File
df = pd.read_csv(csv_path)

# List Image Files in Directory
image_files = [f for f in os.listdir(image_directory) if f.endswith('.jpg')]

# Remove the .jpg extension from the image names
image_names = [os.path.splitext(f)[0] for f in image_files]

# Ensure the DataFrame has enough rows to accommodate all image names
if len(df) < len(image_names):
    additional_rows = len(image_names) - len(df)
    df = pd.concat([df, pd.DataFrame(index=range(additional_rows))], ignore_index=True)

# Fill the 'folder_name' field with the image names
df['folder_name'] = image_names

# Save the updated CSV file
df.to_csv(csv_path, index=False)

# Display the first few rows of the DataFrame to verify the content
print(df.head())