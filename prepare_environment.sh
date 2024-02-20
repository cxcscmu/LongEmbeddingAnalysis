#!/bin/bash

# Check if Conda is installed
if ! command -v conda &> /dev/null; then
    echo "Conda is not installed. Please install Conda and try again."
    exit 1
fi

# Check if the YAML file is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <environment.yaml>"
    exit 1
fi

# Path to the YAML file
yaml_file="$1"

# Create Conda environment
echo "Creating Conda environment from $yaml_file ..."
conda env create -f "$yaml_file"

# Check if environment creation was successful
if [ $? -ne 0 ]; then
    echo "Failed to create Conda environment."
    exit 1
fi

# Activate Conda environment
echo "Activating Conda environment..."
source activate $(head -1 "$yaml_file" | cut -d ' ' -f2)

# Change directory to GradCache if it exists
if [ -d "GradCache" ]; then
    cd GradCache || { echo "Failed to change directory."; exit 1; }
else
    echo "Directory 'GradCache' not found."
    exit 1
fi

pip install .

pip install flash-attn==2.4.2 --no-build-isolation

echo "Installation completed successfully."
cd ..

