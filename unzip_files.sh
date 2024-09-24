#!/bin/bash

SCAN_DIR="/cluster/project/cvg/data/scannet/scans"
OUT_DIR="/cluster/scratch/akjaer/Datasets/ScanNet/scans"
SPLIT_FILE="/cluster/scratch/akjaer/split_files/ONE_train_scene.txt"

# Read the list of scenes from the file into an array
mapfile -t scenes < "$SPLIT_FILE"

for scene in "${scenes[@]}"; do
    # Remove leading/trailing whitespace and carriage return characters
    scene=$(echo "$scene" | tr -d '\r' | xargs)
    
    # Construct the full path to the zip file based on the scene name
    ZIP_FILE="$SCAN_DIR/$scene/${scene}_2d-instance-filt.zip"
    
    # Check if the zip file exists
    if [ -f "$ZIP_FILE" ]; then
        # Unzip the file into its directory
        unzip -o "$ZIP_FILE" -d "$OUT_DIR/$scene/"
        echo "Unzipped $ZIP_FILE in $OUT_DIR/$scene/"
    else
        echo "Zip file $ZIP_FILE not found!"
    fi
done