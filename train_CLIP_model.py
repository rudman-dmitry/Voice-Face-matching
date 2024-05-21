import os
import pickle
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataset import VoiceFaceDataset

import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the Enhanced Projection Network with Adjusted Dimensions and Dropout
class ProjectionNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, output_dim, dropout=0.4):
        super(ProjectionNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.bn1 = nn.BatchNorm1d(hidden_dim_1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.bn2 = nn.BatchNorm1d(hidden_dim_2)
        self.fc3 = nn.Linear(hidden_dim_2, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# Cosine Similarity Function with Temperature Scaling
def cosine_similarity(a, b, temperature=0.06):
    return F.cosine_similarity(a, b, dim=1) / temperature

# Improved Contrastive Loss Function
def contrastive_loss(similarity, labels, margin=0.33):
    positive_loss = labels * F.relu(1 - similarity)
    negative_loss = (1 - labels) * F.relu(similarity - margin)
    return (positive_loss + negative_loss).mean()


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

# Create mappings from indices to keys
v_index_to_key = {i: key for i, key in enumerate(v_embeds.keys())}
f_index_to_key = {i: key for i, key in enumerate(f_embeds.keys())}

# Initialize projection networks with correct input dimensions
image_projection = ProjectionNetwork(input_dim=512, hidden_dim_1=256, hidden_dim_2=128, output_dim=64, dropout=CFG['dropout'])  # Face embeddings
voice_projection = ProjectionNetwork(input_dim=192, hidden_dim_1=164, hidden_dim_2=128, output_dim=64, dropout=CFG['dropout'])  # Voice embeddings

# Plot training statistics
def plot_training_stats(stats, results_folder):
    plt.figure(figsize=(12, 12))
    
    plt.subplot(3, 1, 1)
    plt.plot(stats['train_loss'], label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(stats['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(stats['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Validation Accuracy: {stats["val_acc"][-1]:.3f}')
    plt.legend()

    plt.tight_layout()
    plot_file = os.path.join(results_folder, 'training_stats_CLIP.png')
    plt.savefig(plot_file, dpi=300)
    print(f'Training plot saved to {plot_file}')

# Training Function
def train_model(image_embeddings, voice_embeddings, triplets_train, triplets_val, num_epochs=10, batch_size=48):
    results_folder = 'trained_CLIP_models'
    if not os.path.isdir(results_folder):
        os.makedirs(results_folder)

    # Create mappings from indices to keys
    v_index_to_key = {i: key for i, key in enumerate(voice_embeddings.keys())}
    f_index_to_key = {i: key for i, key in enumerate(image_embeddings.keys())}

    training_data = VoiceFaceDataset(voice_embeddings, image_embeddings, triplets_train, v_index_to_key, f_index_to_key, random_switch_faces=True)
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

    validation_data = VoiceFaceDataset(voice_embeddings, image_embeddings, triplets_val, v_index_to_key, f_index_to_key, random_switch_faces=False)
    val_dataloader = DataLoader(validation_data, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(list(image_projection.parameters()) + list(voice_projection.parameters()), lr=CFG['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=CFG['factor'], patience=CFG['patience'], verbose=True)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        image_projection.cuda()
        voice_projection.cuda()

    best_val_loss = float('inf')
    last_saved = ''
    stats = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        running_loss = 0.0
        image_projection.train()
        voice_projection.train()
        for voice_emb, img_emb1, img_emb2, label in train_dataloader:
            if use_cuda:
                voice_emb, img_emb1, img_emb2, label = voice_emb.cuda(), img_emb1.cuda(), img_emb2.cuda(), label.cuda()

            voice_output = voice_projection(voice_emb)
            img_output1 = image_projection(img_emb1)
            img_output2 = image_projection(img_emb2)

            sim1 = cosine_similarity(voice_output, img_output1, CFG['temperature'])
            sim2 = cosine_similarity(voice_output, img_output2, CFG['temperature'])

            loss = 0.5 * (contrastive_loss(sim1, label, CFG['margin']) + contrastive_loss(sim2, 1 - label, CFG['margin']))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_dataloader)
        stats['train_loss'].append(avg_train_loss)

        # Evaluate on validation set
        val_loss, val_acc = evaluate_model(image_embeddings, voice_embeddings, triplets_val, v_index_to_key, f_index_to_key, return_metrics=True)
        stats['val_loss'].append(val_loss)
        stats['val_acc'].append(val_acc)
        print(f'Epoch {epoch+1}, Loss: {avg_train_loss:.9f}, Val Loss: {val_loss:.9f}, Val Accuracy: {val_acc:.4f}%, LR: {optimizer.param_groups[0]["lr"]:.7f}')

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(results_folder, f'best_model_epoch{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'image_model_state_dict': image_projection.state_dict(),
                'voice_model_state_dict': voice_projection.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }, best_model_path)
            if os.path.exists(last_saved):
                os.remove(last_saved)
            last_saved = best_model_path
            print(f'Saved best model to {best_model_path}')

    final_model_path = os.path.join(results_folder, 'final_model.pth')
    torch.save({
        'epoch': epoch,
        'image_model_state_dict': image_projection.state_dict(),
        'voice_model_state_dict': voice_projection.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_train_loss,
    }, final_model_path)
    print(f'Final model saved to {final_model_path}')

    plot_training_stats(stats, results_folder)

# Evaluation Function
def evaluate_model(image_embeddings, voice_embeddings, triplets_test, v_index_to_key, f_index_to_key, return_metrics=False):
    test_data = VoiceFaceDataset(voice_embeddings, image_embeddings, triplets_test, v_index_to_key, f_index_to_key, random_switch_faces=True)
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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    num_epochs = 5
    train_model(f_embeds, v_embeds, triplets_train, triplets_val, num_epochs)
    evaluate_model(f_embeds, v_embeds, triplets_test, v_index_to_key, f_index_to_key)

    head1 = ProjectionNetwork(input_dim=512, hidden_dim_1=256, hidden_dim_2=128, output_dim=64, dropout=CFG['dropout']) 
    head2 = ProjectionNetwork(input_dim=192, hidden_dim_1=164, hidden_dim_2=128, output_dim=64, dropout=CFG['dropout'])
    total_params = count_parameters(head1) + count_parameters(head2)
    print(f"Total learnable parameters: {total_params}")
