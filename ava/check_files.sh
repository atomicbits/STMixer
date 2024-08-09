#!/bin/bash

# Path to the folders containing video files and folders
video_folder="./videos_15min"
folder_folder="/frames_all"

# List all video files (mp4 and mkv) in the video folder
video_files=$(find "$video_folder" -type f \( -name "*.mp4" -o -name "*.mkv" \) -exec basename {} \;)
echo ${video_files}

# List all folders in the folder folder
folders=$(find "$folder_folder" -type d -exec basename {} \;)

# Loop through each video file and check if a corresponding folder exists
for video_file in $video_files; do
    folder_name="${video_file%.*}"  # Remove the extension to get the folder name
    if ! echo "$folders" | grep -q "$folder_name"; then
        echo "Folder for video '$video_file' not found."
    fi
done