import os
import csv
from pathlib import Path

# Directory to search
# Make sure to use the PCM converted files!!
search_dir = os.path.expanduser("~/STT_workspace/PCM_converted_AUDIO+TEXT_BK")

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
wav_files = []
for root, dirs, files in os.walk(search_dir):
    for file in files:
        if file.endswith(".wav"):
            full_path = os.path.join(root, file)
            wav_files.append(full_path)

# Sort the file names
wav_files.sort()



# Save to CSV file
output_csv = os.path.expanduser("~/STT_workspace/data_filepath_and_scripts.csv")
with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    
    # Write header
    writer.writerow(['File Path'])
    # Write data
    for file_path in wav_files:
        #file_name = os.path.basename(file_path)
        writer.writerow([file_path])

print(f"Total .wav files found: {len(wav_files)}")
print(f"CSV file saved to: {output_csv}")

# Read corresponding .txt files and add content to CSV
output_csv = os.path.expanduser("~/STT_workspace/data_filepath_and_scripts.csv")
with open(output_csv, 'w', newline='', encoding='euc-kr') as csvfile:
    writer = csv.writer(csvfile)
    
    # Write header
    writer.writerow(['File_Path', 'Text_Content'])
    # Write data
    for file_path in wav_files:
        #file_name = os.path.basename(file_path)
        txt_path = file_path.rsplit('.', 1)[0] + '.txt' # converts the .pcm filepath name to end as ".txt"  to get the txt script contents. This also ensures we have the same order of the files.

        text_content = ""
        if os.path.exists(txt_path):
            with open(txt_path, 'r', encoding='euc-kr') as txtfile: # EUC-KR (Extended Unix Code for Korean) or CP949 (Windows Korean encoding), which are common for Korean text files
                text_content = txtfile.read()
        
        writer.writerow([file_path, text_content])

    