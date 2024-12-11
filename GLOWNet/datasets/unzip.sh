#!/bin/bash

# Define the directory where the DIV2K dataset is located
dataset_dir="DIV2K"  # Replace with the actual path to your DIV2K directory

# Define output subdirectories
train_hr_dir="${dataset_dir}/train_HR"
train_lr_dir="${dataset_dir}/train_LR"
valid_hr_dir="${dataset_dir}/valid_HR"
valid_lr_dir="${dataset_dir}/valid_LR"

# Create output directories if they don't exist
mkdir -p "$train_hr_dir" "$train_lr_dir" "$valid_hr_dir" "$valid_lr_dir"

# List of expected zip files for DIV2K
declare -A zip_files
zip_files=(
    ["DIV2K_train_HR.zip"]="$train_hr_dir"
    ["DIV2K_train_LR_bicubic.zip"]="$train_lr_dir"
    ["DIV2K_valid_HR.zip"]="$valid_hr_dir"
    ["DIV2K_valid_LR_bicubic.zip"]="$valid_lr_dir"
)

# Loop through each zip file, unzip, and organize
for zip_file in "${!zip_files[@]}"; do
    zip_path="${dataset_dir}/${zip_file}"
    output_dir="${zip_files[$zip_file]}"

    if [[ -f "$zip_path" ]]; then
        echo "Found $zip_file in $dataset_dir."

        # Check if the directory is empty before unzipping
        if [[ -z "$(ls -A "$output_dir")" ]]; then
            echo "Unzipping $zip_file to $output_dir..."
            unzip -q "$zip_path" -d "$output_dir"
            echo "$zip_file unzipped successfully."
        else
            echo "$output_dir is not empty; skipping extraction for $zip_file."
        fi
    else
        echo "Warning: $zip_file not found in $dataset_dir. Skipping..."
    fi
done

echo "Unzipping process completed."
