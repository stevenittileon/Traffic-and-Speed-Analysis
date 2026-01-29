#!/bin/bash

# Get the directory where the script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check if 'data' directory does not exist and then create it
if [[ ! -e $DIR/data ]]; then
    mkdir "$DIR/data"
else
    echo "'data' directory already exists."
fi

# Download the traffic_analysis.mov file from Google Drive
gdown -O "$DIR/data/traffic_analysis.mp4" "https://drive.google.com/file/d/15pyEEigklnIhqVuycmNWcXCAIxFCFM6b/view?usp=sharing"

# Download the traffic_analysis.pt file from Google Drive
gdown -O "$DIR/data/traffic_analysis.pt" "https://drive.google.com/file/d/11EI7bzOHJYt1pkfeivCEbA7bUFg7x17K/view?usp=sharing"
