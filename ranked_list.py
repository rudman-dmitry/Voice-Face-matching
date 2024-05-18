import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import VoiceFaceDataset
import matplotlib.pyplot as plt
from PIL import Image
from train_CLIP_model import ProjectionNetwork, cosine_similarity

# Function to load the first jpg image from a folder
def load_first_image(name, base_path='vfm_assignment/images'):
    folder_path = os.path.join(base_path, name)
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.jpg'):
                return Image.open(os.path.join(root, file))
    return None

# Load the DataFrames with splits
audio_df_file = 'triplet_data/audio_df_with_splits.pickle'
image_df_file = 'triplet_data/image_df_with_splits.pickle'
with open(audio_df_file, 'rb') as file:
    audio_df = pickle.load(file)
with open(image_df_file, 'rb') as file:
    image_df = pickle.load(file)

# Filter DataFrames for the test set
audio_test_df = audio_df[audio_df['split'] == 'test']
image_test_df = image_df[image_df['split'] == 'test']

# Extract the embeddings and names from the DataFrames
v_embeds_array = np.stack(audio_test_df['embeddings'].values)
f_embeds_array = np.stack(image_test_df['embeddings'].values)
v_nms = audio_test_df['name'].values
f_nms = image_test_df['name'].values

# Initialize projection networks with correct input dimensions
image_projection = ProjectionNetwork(input_dim=512, hidden_dim_1=256, hidden_dim_2=128, output_dim=64, dropout=0.4)  # Face embeddings
voice_projection = ProjectionNetwork(input_dim=192, hidden_dim_1=164, hidden_dim_2=128, output_dim=64, dropout=0.4)  # Voice embeddings

# Load trained models
image_projection.load_state_dict(torch.load('trained_CLIP_models/final_model.pth')['image_model_state_dict'])
voice_projection.load_state_dict(torch.load('trained_CLIP_models/final_model.pth')['voice_model_state_dict'])

# Set models to evaluation mode
image_projection.eval()
voice_projection.eval()

# Select N random face embeddings from the test set
num_faces = 6
selected_faces_indices = np.random.choice(len(f_embeds_array), num_faces, replace=False)
selected_faces = {f_nms[i]: f_embeds_array[i] for i in selected_faces_indices}

# Ensure the selected face list is not empty
if not selected_faces:
    raise ValueError("No faces selected. Please check the filtering process.")

print(f"Selected faces: {selected_faces.keys()}")

# Ensure the selected voice corresponds to one of the faces by name
selected_face_name = list(selected_faces.keys())[0]
if selected_face_name not in v_nms:
    raise ValueError(f"Voice embedding for {selected_face_name} not found in voice embeddings.")

selected_voice_idx = np.where(v_nms == selected_face_name)[0][0]
selected_voice = v_embeds_array[selected_voice_idx]

# Print the ground truth
print(f"Ground truth: The selected voice embedding corresponds to {selected_face_name}")

# Convert to torch tensors
face_tensors = torch.tensor([selected_faces[k] for k in selected_faces.keys()])
voice_tensor = torch.tensor(selected_voice).unsqueeze(0)  # Add batch dimension

# Project embeddings
with torch.no_grad():
    projected_faces = image_projection(face_tensors)
    projected_voice = voice_projection(voice_tensor)

# Compute cosine similarities
cosine_similarities = cosine_similarity(projected_voice, projected_faces, temperature=0.06).numpy()

# Rank faces based on similarities
ranked_faces = sorted(zip(selected_faces.keys(), cosine_similarities), key=lambda x: x[1], reverse=True)

# Display the ranked list
print("Ranked list of faces based on similarity to the voice embedding:")
for i, (face, similarity) in enumerate(ranked_faces):
    print(f"Rank {i+1}: {face} with similarity {similarity:.4f}")

# Plot the images with similarity scores
plt.figure(figsize=(15, 5))
for i, (face, similarity) in enumerate(ranked_faces):
    image = load_first_image(face)
    if image is not None:
        plt.subplot(1, num_faces, i + 1)
        plt.imshow(image)
        plt.title(f"Rank {i+1}\n{similarity:.4f}")
        plt.axis('off')
plt.show()
