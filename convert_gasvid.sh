#!/bin/bash

# Loop through all mp4 files in the current directory
for file in *.mp4; do
    # Extract the base name without the extension
    base_name="${file%.mp4}"
    
    # Create a directory for the frames
    mkdir -p "$base_name"
    
    # Use ffmpeg to extract frames as PNG images with 5-digit frame IDs
    ffmpeg -i "$file" "$base_name/frame_%05d.png"
done