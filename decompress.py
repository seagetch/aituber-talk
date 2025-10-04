import bz2
import os
import sys

if len(sys.argv) != 2:
    print("Usage: python decompress.py <file_path>")
    sys.exit(1)

file_path = sys.argv[1]
output_path = os.path.splitext(file_path)[0]

try:
    with bz2.BZ2File(file_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            f_out.write(f_in.read())
    print(f"Successfully decompressed {file_path} to {output_path}")
    os.remove(file_path)
    print(f"Successfully removed {file_path}")
except Exception as e:
    print(f"An error occurred: {e}")