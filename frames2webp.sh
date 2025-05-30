#!/bin/bash

INPUT_DIR="gasvid_res_full"

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Directory $INPUT_DIR does not exist."
    exit 1
fi

# Loop through all directories in gasvid_res_full
for VIDEO_NAME in "$INPUT_DIR"/*/; do
    VIDEO_NAME=$(basename "$VIDEO_NAME")
    OUTPUT_FILE="${INPUT_DIR}/${VIDEO_NAME}.webp"

    # Combine PNGs into a single WEBP file
    echo "Combining PNGs from ${INPUT_DIR}/${VIDEO_NAME} into $OUTPUT_FILE..."
    if ! ffmpeg -y -framerate 5 -i "${INPUT_DIR}/${VIDEO_NAME}/%*.png" -vcodec libwebp -lossless 1 -loop 0 "$OUTPUT_FILE"; then
        echo "Error: Failed to create WEBP file for $VIDEO_NAME."
        continue
    fi

    echo "WEBP file created successfully: $OUTPUT_FILE"
done
