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

# Check if an environment name is provided
if [[ -n "$1" ]]; then
    ENV_NAME="$1"
else
    ENV_NAME="xr-mujoco"
fi

# Detect the system's default Python version
if command -v python3 &>/dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1)
elif command -v python &>/dev/null; then
    PYTHON_VERSION=$(python --version 2>&1)
else
    echo "Python is not installed on this system."
    exit 1
fi

echo "The system's default Python version is: $PYTHON_VERSION"

# Extract the major and minor version numbers from the Python version string
PYTHON_MAJOR_MINOR=$(echo $PYTHON_VERSION | grep -oP '\d+\.\d+')

# Initialize conda
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    . "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    . "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    echo "Conda initialization script not found. Please install Miniconda or Anaconda."
    exit 1
fi

conda deactivate
conda remove -n "$ENV_NAME" --all -y
conda create -n "$ENV_NAME" python=$PYTHON_MAJOR_MINOR -y

echo "Conda environment '$ENV_NAME' created with Python $PYTHON_MAJOR_MINOR"

# Activate the conda environment
conda activate "$ENV_NAME"

# replace conda c++ dependency with libstdcxx-ng
if [[ "$OS_NAME" == "Linux" ]]; then
    conda install -c conda-forge libstdcxx-ng -y
fi
pip install --upgrade pip

if [[ "$OS_NAME" == "Linux" ]]; then
    # install compilation tools on conda
    conda install -c conda-forge gcc gxx cmake make -y
else
    echo "Unsupported operating system: $OS_NAME"
    exit 1
fi

# Install the required packages

pip install -e .

echo -e "\n"
echo -e "[INFO] Created conda environment named '$ENV_NAME'.\n"
echo -e "To activate the environment, run: conda activate $ENV_NAME"
echo -e "\n"