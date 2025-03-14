#!/bin/bash

# the GitHub release URL
BASE_URL="https://github.com/SJ00425/DL-Project-1-DJT/releases/download/v1.0"

FILES=(
    "PreAct_ResNet10.pth"
    "PreAct_ResNet12.pth"
    "ResNet12.pth"
    "custom_ResNet18_200_epoch.pth"
    "custom_ResNet18_300_epoch.pth"
    "student_custom_ResNet18.pth"
    "student_wideResNet_28-3.7.pth"
    "teacher_PreAct_ResNet152.pth"
    "teacher_wideResNet_40-10.pth"
    "wideResNet_28-3.7.pth"
)

# Download each file
for FILE in "${FILES[@]}"; do
    echo "Downloading $FILE..."
    
    # Use wget or curl depending on availability
    if command -v wget &> /dev/null; then
        wget -q --show-progress "$BASE_URL/$FILE"
    elif command -v curl &> /dev/null; then
        curl -O "$BASE_URL/$FILE"
    else
        echo "Error: wget or curl is required to download files."
        exit 1
    fi
done

echo "All files downloaded successfully into the current directory."
