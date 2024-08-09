#!/usr/bin/env bash

# Script to remove a single video file after processing

# Define the root folder of the files
ROOT_DATA_DIR="./trainval"

# Input video file name (passed as argument)
VIDEO_FILE="$1"

# Ensure the video file exists
if [[ ! -f "${ROOT_DATA_DIR}/${VIDEO_FILE}" ]]; then
  echo "Input video file '${VIDEO_FILE}' not found in '${ROOT_DATA_DIR}'."
  exit 1
fi

# Remove the video file
rm "${ROOT_DATA_DIR}/${VIDEO_FILE}"