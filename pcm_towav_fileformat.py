# pip3 install pydub

from pydub import AudioSegment

# Replace 'input.pcm' with the path to your PCM file
# Replace 'output.wav' with the desired name for your WAV file
# Specify the sample width (e.g., 2 for 16-bit audio)
# Specify the frame rate (e.g., 44100)
# Specify the number of channels (e.g., 1 for mono, 2 for stereo)

audio = AudioSegment.from_file("LIST00000_F_YJS00_41_수도권_실내_00007.pcm", format="raw",
                                 sample_width=2, frame_rate=44100, channels=1)
audio.export("output.wav", format="wav")

print("Conversion complete!")

import os
from pydub import AudioSegment

# Replace this with the directory where you upload your files in Colab
# For example, if you upload to a folder named 'my_audio_files', use '/content/my_audio_files'
input_directory = "/content/" # Assuming files are in the content directory after upload

# Create an output directory if it doesn't exist
output_directory = "/content/converted_wavs"
os.makedirs(output_directory, exist_ok=True)

for filename in os.listdir(input_directory):
    if filename.endswith(".pcm"):
        input_filepath = os.path.join(input_directory, filename)
        output_filename = os.path.splitext(filename)[0] + ".wav"
        output_filepath = os.path.join(output_directory, output_filename)

        try:
            audio = AudioSegment.from_file(input_filepath, format="raw",
                                         sample_width=2, frame_rate=44100, channels=1)
            audio.export(output_filepath, format="wav")
            print(f"Converted {filename} to {output_filename}")
        except Exception as e:
            print(f"Error converting {filename}: {e}")

print("Conversion process complete!")