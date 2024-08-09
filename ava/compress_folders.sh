#!/bin/bash

# Define the parent directory containing the folders with image files
PARENT_DIR="./frames"

# Define a directory to store the compressed files (outside the parent directory)
COMPRESSED_DIR="./frames_compressed"
mkdir -p "$COMPRESSED_DIR"

# Loop through each folder in the parent directory
for folder in "$PARENT_DIR"/*; do
    if [ -d "$folder" ]; then
        folder_name=$(basename "$folder")
        archive="$COMPRESSED_DIR/$folder_name.tar.gz"
        if [ -f "$archive" ]; then
            echo "Skipping $folder_name, archive already exists."
        else
            tar -czf "$archive" -C "$PARENT_DIR" -- "$folder_name"
            echo "Compressed $folder_name"
        fi
    fi
done

echo "All folders processed."
