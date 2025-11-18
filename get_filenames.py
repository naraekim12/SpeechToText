import os
import csv
from pathlib import Path

# Directory to search
search_dir = os.path.expanduser("~/STT_workspace/AUDIO+TEXT_BK")

# Change Korean filenames to English 
import os, glob, re

name_map = {
    "수도권_실내": "Metropolitan_indoors",
    "수도권_녹음실": "Metropolitan_recordingbooth"
}

for root, dirs, files in os.walk(search_dir):
    for f in files:
        for name in name_map.keys():
            if re.search(name,f) != None:
                new_name = re.sub(name,name_map[name],f)
                try:
                    os.rename(os.path.join(root,f), os.path.join(root, new_name))
                except OSError:
                    print("No such file or directory!")


# Get all .pcm files recursively
pcm_files = []
for root, dirs, files in os.walk(search_dir):
    for file in files:
        if file.endswith(".pcm"):
            full_path = os.path.join(root, file)
            pcm_files.append(full_path)

# Sort the file names
pcm_files.sort()

# Save to CSV file
output_csv = os.path.expanduser("~/STT_workspace/data_filepath_and_scripts.csv")
with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    
    # Write header
    writer.writerow(['File Path', 'File Name'])
    # Write data
    for file_path in pcm_files:
        file_name = os.path.basename(file_path)
        writer.writerow([file_path, file_name])

print(f"Total .pcm files found: {len(pcm_files)}")
print(f"CSV file saved to: {output_csv}")

