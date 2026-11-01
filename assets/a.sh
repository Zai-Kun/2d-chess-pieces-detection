#!/usr/bin/env bash
# Extracts every *.zip in the current directory, grabs only the
# diffuse/albedo/color map from each (we don't need normal, ao,
# displacement, or specular maps for 2D compositing), and drops a
# cleanly-named copy into ./picked/. Leaves originals untouched.
#
# Usage: ./extract_textures.sh
# Optional: RESIZE=1024 ./extract_textures.sh   -> also downscale picks
#           (requires imagemagick's `convert`/`magick`)

set -euo pipefail

PICKED_DIR="picked"
WORK_DIR=".extract_tmp"
RESIZE="${RESIZE:-}"

mkdir -p "$PICKED_DIR"
rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"

shopt -s nullglob
zips=(*.zip)
shopt -u nullglob

if [ ${#zips[@]} -eq 0 ]; then
    echo "No .zip files found in $(pwd)"
    exit 1
fi

for zip in "${zips[@]}"; do
    name="${zip%.zip}"
    dest_dir="$WORK_DIR/$name"
    mkdir -p "$dest_dir"

    echo "Extracting: $zip"
    if ! 7z x -y -o"$dest_dir" "$zip" > /dev/null; then
        echo "  Failed to extract $zip, skipping."
        continue
    fi

    # Find the diffuse/albedo/color map, case-insensitive, ignoring
    # normal/ao/displacement/specular/roughness/metalness/height/bump maps.
    match=$(find "$dest_dir" -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" \) \
        | grep -Ei 'diffuse|albedo|_color|basecolor' \
        | grep -Eiv 'normal|_ao|ambient|displace|specular|rough|metal|height|bump' \
        | head -n 1) || true

    if [ -z "$match" ]; then
        echo "  No diffuse/albedo map found for $name, skipping."
        continue
    fi

    ext="${match##*.}"
    out_path="$PICKED_DIR/${name}.${ext}"
    cp "$match" "$out_path"
    echo "  Picked: $(basename "$match") -> $out_path"

    if [ -n "$RESIZE" ]; then
        if command -v magick > /dev/null 2>&1; then
            magick "$out_path" -resize "${RESIZE}x${RESIZE}" "$out_path"
        elif command -v convert > /dev/null 2>&1; then
            convert "$out_path" -resize "${RESIZE}x${RESIZE}" "$out_path"
        else
            echo "  RESIZE requested but imagemagick not found, skipping resize."
        fi
    fi
done

rm -rf "$WORK_DIR"

echo ""
echo "Done. Picked textures are in ./$PICKED_DIR/"
ls -la "$PICKED_DIR"
