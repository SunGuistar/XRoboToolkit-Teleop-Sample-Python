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
elif [[ "$OS_NAME" == "MINGW"* || "$OS_NAME" == "CYGWIN"* || "$OS_NAME" == "MSYS"* ]]; then
    OS_VERSION="Windows"
    echo "Windows detected"
else
    echo "Unsupported operating system: $OS_NAME"
    exit 1
fi

echo "Operating system check passed: $OS_NAME $OS_VERSION"

    # Install the required packages
    rm -rf dependencies
    mkdir dependencies
    cd dependencies
    git clone https://github.com/XR-Robotics/XRoboToolkit-PC-Service.git
    cd XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK 
    bash build.sh || { echo "Failed to build PXREARobotSDK"; exit 1; }
    cd ../../..

    git clone https://github.com/XR-Robotics/XRoboToolkit-PC-Service-Pybind.git
    cd XRoboToolkit-PC-Service-Pybind
    if [[ "$OS_NAME" == "Linux" ]]; then
        cp ../XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/build/libPXREARobotSDK.so lib/ || { echo "Failed to copy libPXREARobotSDK.so"; exit 1; }
    elif [[ "$OS_NAME" == "MINGW"* || "$OS_NAME" == "CYGWIN"* || "$OS_NAME" == "MSYS"* ]]; then
        cp ../XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/build/PXREARobotSDK.dll lib/ || { echo "Failed to copy PXREARobotSDK.dll"; exit 1; }
        cp ../XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/build/PXREARobotSDK.lib lib/ || { echo "Failed to copy PXREARobotSDK.lib"; exit 1; }
    fi
    python setup.py install || { echo "Failed to install xrobotoolkit_sdk"; exit 1; }
    cd ..
    rm -rf XRoboToolkit-PC-Service

    cd ..

    pip install -e . || { echo "Failed to install xrobotoolkit_teleop with pip"; exit 1; }


    echo -e "\n"
    echo -e "[INFO] xrobotoolkit_teleop is installed in conda environment '$ENV_NAME'.\n"
    echo -e "\n"
