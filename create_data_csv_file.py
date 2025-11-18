import argparse
import csv
from pathlib import Path
import sys

# #!/usr/bin/env python3
# """
# create_data_csv_file.py

# List all files in a specified folder (optionally recursive), read text from .txt files,
# and save filepath + text to a CSV in the same order files are discovered.
# """


def collect_filepaths(folder: Path, recursive: bool):
    if recursive:
        iterator = folder.rglob('*')
    else:
        iterator = folder.iterdir()
    for p in iterator:
        try:
            if p.is_file():
                yield p
        except PermissionError:
            continue

def main():
    parser = argparse.ArgumentParser(description="Save filepaths and text (for .txt) from a folder into a CSV.")
    parser.add_argument('--folder', '-f', type=Path, default=Path('.'),
                        help='Folder to scan (default: current directory)')
    parser.add_argument('--output', '-o', type=Path, default=Path('filepaths.csv'),
                        help='Output CSV file (default: filepaths.csv)')
    parser.add_argument('--no-recursive', action='store_true',
                        help='Do not traverse subdirectories')
    parser.add_argument('--relative', action='store_true',
                        help='Save relative paths instead of absolute')
    args = parser.parse_args()

    folder = args.folder.expanduser().resolve()
    if not folder.exists() or not folder.is_dir():
        print(f"Error: folder {folder} does not exist or is not a directory.", file=sys.stderr)
        sys.exit(1)

    files = list(collect_filepaths(folder, recursive=not args.no_recursive))

    out_path = args.output
    try:
        with out_path.open('w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['filepath', 'text'])
            for f in files:
                try:
                    path_str = str(f.relative_to(folder)) if args.relative else str(f.resolve())
                except Exception:
                    # fallback if relative_to fails
                    path_str = str(f.resolve())
                text = ''
                if f.suffix.lower() == '.txt':
                    try:
                        # read text; replace errors to avoid decoding issues
                        text = f.read_text(encoding='utf-8', errors='replace')
                    except Exception as e:
                        text = f"<ERROR_READING_FILE: {e}>"
                writer.writerow([path_str, text])
    except Exception as e:
        print(f"Error writing CSV: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()


# import os, csv

# current_file_dir = os.path.dirname(os.path.abspath(__file__))
# print(current_file_dir)

# f=open("/home/narae/STT_workspace/data_filepath_and_scripts.csv",'r+')
# w=csv.writer(f)
# for path, dirs, files in os.walk("/home/narae/STT_workspace/AUDIO+TEXT_BK"):
#     for filename in files:
#         w.writerow([filename])