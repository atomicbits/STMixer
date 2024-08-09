#!/usr/bin/env bash

# Script to process a single video file

# Define the root folder of the files
ROOT_DATA_DIR="./trainval"

# Check if the output directory exists, if not create it
OUT_DATA_DIR="./videos_15min"
if [[ ! -d "${OUT_DATA_DIR}" ]]; then
  echo "${OUT_DATA_DIR} doesn't exist. Creating it."
  mkdir -p "${OUT_DATA_DIR}"
fi

# Input video file name (passed as argument)
VIDEO_FILE="$1"

# Ensure the video file exists
if [[ ! -f "${ROOT_DATA_DIR}/${VIDEO_FILE}" ]]; then
  echo "Input video file '${VIDEO_FILE}' not found in '${ROOT_DATA_DIR}'."
  exit 1
fi

# Output file name
OUT_NAME="${OUT_DATA_DIR}/${VIDEO_FILE}"

# Check if the output file already exists, if not process the video
if [ ! -f "${OUT_NAME}" ]; then
  ffmpeg -ss 900 -t 901 -i "${ROOT_DATA_DIR}/${VIDEO_FILE}" "${OUT_NAME}"
fi
