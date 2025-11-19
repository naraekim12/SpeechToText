# # --------------- Run this for a single file conversion ---------------

# # pip3 install pydub
# from pydub import AudioSegment

# # Replace 'input.pcm' with the path to your PCM file
# # Replace 'output.wav' with the desired name for your WAV file
# # Specify the sample width (e.g., 2 for 16-bit audio)
# # Specify the frame rate (e.g., 44100)
# # Specify the number of channels (e.g., 1 for mono, 2 for stereo)

# audio = AudioSegment.from_file("LIST00000_F_YJS00_41_수도권_실내_00007.pcm", format="raw",
#                                  sample_width=2, frame_rate=44100, channels=1)
# audio.export("output.wav", format="wav")

# print("Conversion complete!")

# import os
# from pydub import AudioSegment

# # Replace this with the directory where you upload your files in Colab
# # For example, if you upload to a folder named 'my_audio_files', use '/content/my_audio_files'
# input_directory = "/content/" # Assuming files are in the content directory after upload

# # Create an output directory if it doesn't exist
# output_directory = "/content/converted_wavs"
# os.makedirs(output_directory, exist_ok=True)

# for filename in os.listdir(input_directory):
#     if filename.endswith(".pcm"):
#         input_filepath = os.path.join(input_directory, filename)
#         output_filename = os.path.splitext(filename)[0] + ".wav"
#         output_filepath = os.path.join(output_directory, output_filename)

#         try:
#             audio = AudioSegment.from_file(input_filepath, format="raw",
#                                          sample_width=2, frame_rate=44100, channels=1)
#             audio.export(output_filepath, format="wav")
#             print(f"Converted {filename} to {output_filename}")
#         except Exception as e:
#             print(f"Error converting {filename}: {e}")

# print("Conversion process complete!")



# # --------------- Run this for a batch file conversion --------------------



# pip3 install pydub
# Requires system ffmpeg installed (sudo apt install ffmpeg) or ffmpeg on PATH (Windows)

import argparse
import subprocess
from pathlib import Path

def convert_with_ffmpeg(input_path: Path, output_path: Path, pcm_rate: int, pcm_channels: int, pcm_format: str):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if input_path.suffix.lower() == ".pcm":
        # raw PCM -> tell ffmpeg the format (s16le for 16-bit little-endian). adjust pcm_format if needed.
        cmd = [
            "ffmpeg", "-y",
            "-f", pcm_format,
            "-ar", str(pcm_rate),
            "-ac", str(pcm_channels),
            "-i", str(input_path),
            str(output_path)
        ]
    else:
        # any other audio file -> let ffmpeg auto-detect
        cmd = ["ffmpeg", "-y", "-i", str(input_path),
               "-ar", str(pcm_rate), "-ac", str(pcm_channels), str(output_path)]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"Converted: {input_path} -> {output_path}")
    except subprocess.CalledProcessError:
        print(f"Failed to convert: {input_path}")

def batch_convert(input_dir: Path, output_dir: Path, pcm_rate: int, pcm_channels: int, pcm_format: str, recurse: bool):
    patterns = ["*.pcm", "*.wav", "*.mp3", "*.m4a", "*.flac", "*.ogg", "*.aac", "*.wma"]
    if recurse:
        files = [p for pat in patterns for p in input_dir.rglob(pat)]
    else:
        files = [p for pat in patterns for p in input_dir.glob(pat)]
    for inp in sorted(files):
        # skip if already a wav in output (or skip converting wav->wav)
        rel = inp.relative_to(input_dir)
        out = output_dir.joinpath(rel).with_suffix(".wav")
        convert_with_ffmpeg(inp, out, pcm_rate, pcm_channels, pcm_format)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch convert audio files to WAV using ffmpeg")
    parser.add_argument("input_dir", help="Input directory (use quotes if path has spaces)")
    parser.add_argument("output_dir", help="Output directory to store WAVs")
    parser.add_argument("--rate", type=int, default=44100, help="Target sample rate (default: 44100)")
    parser.add_argument("--channels", type=int, default=1, help="Target channels (1=mono, 2=stereo) (default:1)")
    parser.add_argument("--pcm-format", default="s16le", help="PCM format for .pcm files (default: s16le for 16-bit LE)")
    parser.add_argument("--no-recurse", action="store_true", help="Do not recurse into subfolders")
    args = parser.parse_args()

    inp = Path(args.input_dir)
    outp = Path(args.output_dir)
    if not inp.exists():
        raise SystemExit(f"Input directory not found: {inp}")

    batch_convert(inp, outp, args.rate, args.channels, args.pcm_format, recurse=not args.no_recurse)
    print("All done.")
    print(f"Converted files are in: {outp}")
    print("Total number of file: ", len(list(outp.rglob("*.wav"))))