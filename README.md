# Voice-Face Matching

## Overview

This project develops a system that matches voice embeddings with corresponding face embeddings. It leverages a neural network model to determine the likelihood of a voice matching one of two faces.

## Setup

### Prerequisites

- Docker
- Python 3.9

### Instructions

1. **Clone the repository**:

    ```bash
    git clone https://github.com/rudman-dmitry/voice-face-matching.git
    cd voice-face-matching
    ```

2. **Download the data**:

    Run the following command to download and unzip the data:

    ```bash
    python download_data.py
    ```

3. **Prepare the data**:

    Run the following command to prepare the data:

    ```bash
    python prepare_data.py
    ```

4. **Build the Docker image**:

    ```bash
    docker build -t voice-face-matching:latest .
    ```

5. **Run the Docker container**:

    ```bash
    docker run -p 8888:8888 -v $(pwd):/app -it voice-face-matching:latest
    ```

    This command maps port 8888 of the Docker container to port 8888 on your host machine and mounts the current directory into the container at `/app`.

6. **Access Jupyter Notebook**:

    Open your web browser and navigate to the URL provided in the terminal to access Jupyter Notebook.

## Project Structure

- `download_data.py`: Script to download and unzip the dataset.
- `prepare_data.py`: Script to prepare the data for training.
- `dataset.py`: Custom dataset class for handling voice and face embeddings.
- `train_CLIP_model.py`: Training script for the CLIP model.
- `train_VFTC_model.py`: Training script for the VoiceFaceTripletsClassifier model.
- `test_CLIP.py`: Testing script for the CLIP model.
- `test_VFTC.py`: Testing script for the VoiceFaceTripletsClassifier model.
- `Data_exploration_and_training.ipynb`: Jupyter Notebook for data exploration and training.

## Requirements

The required Python packages are listed in `requirements.txt`.

## Notes

- Ensure Docker is installed and running on your system.
- Follow the instructions carefully to set up the environment and run the project.
