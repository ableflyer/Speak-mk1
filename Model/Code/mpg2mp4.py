import os
import subprocess
import sys

def convert_mpg_to_mp4(source_folder):
    # Ensure path exists
    if not os.path.exists(source_folder):
        print(f"Error: Folder '{source_folder}' not found.")
        return

    # Get list of all mpg files
    files = [f for f in os.listdir(source_folder) if f.lower().endswith('.mpg')]
    
    if not files:
        print("No .mpg files found in the directory.")
        return

    print(f"Found {len(files)} MPG files to convert.")

    for filename in files:
        input_path = os.path.join(source_folder, filename)
        # Change extension to .mp4
        output_path = os.path.join(source_folder, os.path.splitext(filename)[0] + '.mp4')

        print(f"Converting: {filename} -> {os.path.basename(output_path)}")

        # FFmpeg command
        # -i: input file
        # -c:v libx264: use H.264 codec (standard for mp4)
        # -preset fast: faster conversion
        # -c:a aac: convert audio to AAC (standard for mp4)
        # -y: overwrite output file if exists
        cmd = [
            "ffmpeg",
            "-i", input_path,
            "-c:v", "libx264",
            "-preset", "fast",
            "-c:a", "aac",
            "-y",
            output_path
        ]

        try:
            # Run conversion and hide detailed output
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Delete original file ONLY if conversion succeeded
            os.remove(input_path)
            print(f"  ✔ Converted and deleted original: {filename}")

        except subprocess.CalledProcessError as e:
            print(f"  ✖ Failed to convert {filename}. Keeping original file.")
            print(f"     Error: {e}")

if __name__ == "__main__":
    # Use current directory if no argument is passed, otherwise use the passed path
    target_folder = "../Data/GRID_Dataset/s1"
    
    print(f"Working in: {target_folder}")
    convert_mpg_to_mp4(target_folder)