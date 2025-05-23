#!/bin/bash

# Check the operating system
OS_NAME=$(uname -s)
OS_VERSION=""

if [[ "$OS_NAME" == "Linux" ]]; then
    if command -v lsb_release &>/dev/null; then
        OS_VERSION=$(lsb_release -rs)
    elif [[ -f /etc/os-release ]]; then
        . /etc/os-release
        OS_VERSION=$VERSION_ID
    fi
    if [[ "$OS_VERSION" != "22.04" && "$OS_VERSION" != "24.04" ]]; then
        echo "This script only supports Ubuntu 22.04 or 24.04."
        exit 1
    fi
else
    echo "Unsupported operating system: $OS_NAME"
    exit 1
fi

echo "Operating system check passed: $OS_NAME $OS_VERSION"

# Install the required packages
rm -rf dependencies
mkdir dependencies
cd dependencies
git clone https://github.com/XR-Robotics/RoboticsService-PC.git
cd RoboticsService-PC/RoboticsService/PXREARobotSDK 
bash build.sh || { echo "Failed to build PXREARobotSDK"; exit 1; }
cd ../../..

git clone https://github.com/XR-Robotics/RoboticsService-Python.git
cd RoboticsService-Python
cp ../RoboticsService-PC/RoboticsService/PXREARobotSDK/build/libPXREARobotSDK.so lib/ || { echo "Failed to copy libPXREARobotSDK.so"; exit 1; }
python setup.py install || { echo "Failed to install RoboticsService-Python"; exit 1; }

# cd ..
# git clone https://github.com/zhigenzhao/placo.git
# cd placo
# git checkout xml_experimental || { echo "Failed to checkout xml_experimental branch"; exit 1; }
# bash setup.sh --install || { echo "Failed to run placo setup"; exit 1; }

cd ../..

pip install -e . || { echo "Failed to install teleop_demo_mujoco with pip"; exit 1; }
