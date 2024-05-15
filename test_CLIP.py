import os
import pickle
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataset import VoiceFaceDataset
import torch.nn as nn
import torch.nn.functional as F
from train_CLIP_model import ProjectionNetwork, cosine_similarity, contrastive_loss

# Configuration parameters
CFG = dict(
    dropout=0.4,
    temperature=0.06,
    margin=0.33,
    learning_rate=0.0001,
    patience=35,
    factor=0.9
)

# Load the data file
data_file = 'triplet_data/triplets_1_negatives.pickle'
with open(data_file, 'rb') as file:
    data = pickle.load(file)

# Assign the loaded data to respective variables
v_embeds, v_nms, f_embeds, f_nms, triplets, triplets_train, triplets_val, triplets_test = data

# Initialize projection networks with correct input dimensions
image_projection = ProjectionNetwork(input_dim=512, hidden_dim_1=256, hidden_dim_2=128, output_dim=64, dropout=CFG['dropout'])  # Face embeddings
voice_projection = ProjectionNetwork(input_dim=192, hidden_dim_1=164, hidden_dim_2=128, output_dim=64, dropout=CFG['dropout'])  # Voice embeddings

# Load pre-trained model weights
checkpoint = torch.load('trained_CLIP_models/final_model.pth')
image_projection.load_state_dict(checkpoint['image_model_state_dict'])
voice_projection.load_state_dict(checkpoint['voice_model_state_dict'])

# Evaluation Function
def evaluate_model(image_embeddings, voice_embeddings, triplets_test, return_metrics=False):
    test_data = VoiceFaceDataset(voice_embeddings, image_embeddings, triplets_test, random_switch_faces=True)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        image_projection.cuda()
        voice_projection.cuda()

    correct = 0
    total = len(test_dataloader)
    total_loss = 0
    image_projection.eval()
    voice_projection.eval()
    with torch.no_grad():
        for voice_emb, img_emb1, img_emb2, label in test_dataloader:
            if use_cuda:
                voice_emb, img_emb1, img_emb2, label = voice_emb.cuda(), img_emb1.cuda(), img_emb2.cuda(), label.cuda()

            voice_output = voice_projection(voice_emb)
            img_output1 = image_projection(img_emb1)
            img_output2 = image_projection(img_emb2)

            sim1 = cosine_similarity(voice_output, img_output1, CFG['temperature'])
            sim2 = cosine_similarity(voice_output, img_output2, CFG['temperature'])

            loss = contrastive_loss(sim1, label, CFG['margin']) + contrastive_loss(sim2, 1 - label, CFG['margin'])
            total_loss += loss.item()
            prediction = (sim1 >= 0.5).float()
            correct += (prediction == label).sum().item()

    accuracy = correct / total * 100
    avg_loss = total_loss / total
    if return_metrics:
        return avg_loss, accuracy
    print(f'Identification Accuracy: {accuracy:.2f}%')

if __name__ == "__main__":
    evaluate_model(f_embeds, v_embeds, triplets_test)
