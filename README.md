# Voice-Face Matching

## Overview

This project develops a system to match voice embeddings with corresponding face embeddings using two different models:

1. **Voice-Face-Triplet-Classifier (VFTC)**: This model implements the static approach described in the paper ["Seeing Voices and Hearing Faces"](https://arxiv.org/pdf/1804.00326). It uses triplet loss to train the network to distinguish between matching and non-matching voice-face pairs.

2. **CLIP Adaptation for Voice-Face Matching**: This model adapts the Contrastive Language-Image Pre-Training (CLIP) framework for the voice-face matching task, leveraging its robust multimodal capabilities. The approach is based on techniques described in the paper ["Learning Transferable Visual Models From Natural Language Supervision"](https://doi.org/10.48550/arXiv.2103.00020).

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

2. **Build the Docker image**:

    ```bash
    docker build -t voice-face-matching:latest .
    ```

3. **Download the data**:

    Run the following command to download and unzip the data:

    ```bash
    python download_data.py
    ```

4. **Prepare the data**:

    Run the following command to prepare the data:

    ```bash
    python prepare_data.py
    ```

5. **Run the Docker container**:

    ```bash
    docker run -v $(pwd):/app -it --name voice-face-matching-container voice-face-matching:latest
    ```

    This command mounts the current directory into the container at `/app` and starts an interactive terminal session.

6. **Run the scripts inside the Docker container**:

    Use the following commands to run the necessary scripts inside the Docker container. These commands assume you have an active terminal session within the container.

    - **Prepare the Data**:
        ```bash
        python /app/prepare_data.py
        ```

    - **Train the Voice-Face Triplets Classifier Model**:
        ```bash
        python /app/train_VFTC_model.py
        ```

    - **Test the Voice-Face Triplets Classifier Model**:
        ```bash
        python /app/test_VFTC.py
        ```

    - **Train the CLIP Model**:
        ```bash
        python /app/train_CLIP_model.py
        ```

    - **Test the CLIP Model**:
        ```bash
        python /app/test_CLIP.py
        ```

    - **Generate Ranked List of Faces**:
        ```bash
        python /app/ranked_list.py
        ```

### Alternative Method: Running Commands Directly in the Container

If you prefer not to keep an interactive terminal session open, you can use `docker exec` to run commands in the container after it is started.

1. **Start the Docker container in detached mode**:

    ```bash
    docker run -v $(pwd):/app -d --name voice-face-matching-container voice-face-matching:latest
    ```

2. **Run the scripts using `docker exec`**:

    - **Prepare the Data**:
        ```bash
        docker exec -it voice-face-matching-container python /app/prepare_data.py
        ```

    - **Train the Voice-Face Triplets Classifier Model**:
        ```bash
        docker exec -it voice-face-matching-container python /app/train_VFTC_model.py
        ```

    - **Test the Voice-Face Triplets Classifier Model**:
        ```bash
        docker exec -it voice-face-matching-container python /app/test_VFTC.py
        ```

    - **Train the CLIP Model**:
        ```bash
        docker exec -it voice-face-matching-container python /app/train_CLIP_model.py
        ```

    - **Test the CLIP Model**:
        ```bash
        docker exec -it voice-face-matching-container python /app/test_CLIP.py
        ```

    - **Generate Ranked List of Faces**:
        ```bash
        docker exec -it voice-face-matching-container python /app/ranked_list.py
        ```

## Project Structure

- `download_data.py`: Script to download and unzip the dataset.
- `prepare_data.py`: Script to prepare the data for training.
- `dataset.py`: Custom dataset class for handling voice and face embeddings.
- `ranked_list.py`: Utilizes the trained CLIP model weights to produce a ranked list of faces based on similarity score.
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
