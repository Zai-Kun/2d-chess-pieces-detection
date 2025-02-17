import os
import cairosvg

# Define source and target directories
source_dir = "raw_pieces_lichess"
output_dir = "pieces"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Loop through each subdirectory in the source directory
for subdir in os.listdir(source_dir):
    subdir_path = os.path.join(source_dir, subdir)
    target_subdir = os.path.join(output_dir, subdir)

    if os.path.isdir(subdir_path):
        os.makedirs(target_subdir, exist_ok=True)

        for filename in os.listdir(subdir_path):
            if filename.endswith(".svg"):
                svg_path = os.path.join(subdir_path, filename)
                png_path = os.path.join(target_subdir, filename.replace(".svg", ".png"))

                print(f"Converting {svg_path} to PNG...")

                # Convert SVG to PNG using CairoSVG (300x300, keeping transparency)
                cairosvg.svg2png(url=svg_path, write_to=png_path, output_width=300, output_height=300)

print("Conversion completed.")
