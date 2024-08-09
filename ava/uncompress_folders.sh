#!/bin/bash

# Define the directory containing the compressed files
COMPRESSED_DIR="/data_compressed/frames_compressed"
# Define the directory where you want to extract the files
EXTRACT_DIR="/data/ava/frames"

# Create the extract directory if it doesn't exist
mkdir -p "$EXTRACT_DIR"

# Loop through each compressed file in the compressed directory
for archive in "$COMPRESSED_DIR"/*.tar.gz; do
    if [ -f "$archive" ]; then
        tar -xzf "$archive" -C "$EXTRACT_DIR"
        echo "Uncompressed $(basename "$archive")"
    fi
done

echo "All archives uncompressed."
