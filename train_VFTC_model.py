import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import time
from dataset import VoiceFaceDataset

# Configuration parameters
CFG = dict(input_layer_size=256, dropout=0.5, loss_fn=nn.BCEWithLogitsLoss())

# Define the Triplets Classifier Model
class VoiceFaceTripletsClassifier(nn.Module):
    """
    A neural network model for classifying triplets of voice and face embeddings to determine whether a given voice matches one of two faces.

    This classifier takes the embeddings of a voice, a matching face, and a non-matching face, and outputs a score indicating the likelihood that the voice matches the first face more than the second.

    Attributes:
        fc1 (nn.Linear): First fully connected layer that expands the concatenated embeddings.
        fc2 (nn.Linear): Second fully connected layer that reduces dimensionality.
        fc3 (nn.Linear): Final fully connected layer that outputs the classification score.
        pool (nn.MaxPool1d): Max pooling layer to reduce dimensionality between layers.
        dropout (nn.Dropout): Dropout layer for regularization to prevent overfitting.
        batch_norm1 (nn.BatchNorm1d): Batch normalization layer to standardize inputs during training.

    Parameters:
        input_sz_voice (int): Size of the voice embedding vector.
        input_sz_face (int): Size of the face embedding vector.
        cfg (dict): Configuration dictionary containing:
            - 'input_layer_size' (int): Size of the input layer for the first fully connected layer.
            - 'dropout' (float): Dropout rate for regularization.

    The architecture concatenates the voice and face embeddings and processes them through three linear layers with ReLU activations, interspersed with max pooling, dropout, and batch normalization to enhance learning stability and performance.
    """
    def __init__(self, input_sz_voice, input_sz_face, cfg):
        super().__init__()
        input_layer_size = cfg['input_layer_size']
        dropout = cfg['dropout']
        dim = input_sz_voice + 2 * input_sz_face
        self.fc1 = nn.Linear(dim, input_layer_size)
        self.fc2 = nn.Linear(input_layer_size // 2, input_layer_size // 4)
        self.fc3 = nn.Linear(input_layer_size // 8, 1)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(dropout)
        self.batch_norm1 = nn.BatchNorm1d(num_features=input_layer_size // 2)

    def forward(self, x_v, x_f1, x_f2):
        """
        Forward pass through the network.

        Parameters:
            x_v (torch.Tensor): Tensor containing the voice embeddings.
            x_f1 (torch.Tensor): Tensor containing the first face embeddings (matching face).
            x_f2 (torch.Tensor): Tensor containing the second face embeddings (non-matching face).

        Returns:
            torch.Tensor: Output tensor with the classification scores for each triplet.
        """
        x = torch.cat((x_v, x_f1, x_f2), dim=1)
        x = self.pool(torch.relu(self.fc1(x)))
        x = self.batch_norm1(x)
        x = self.dropout(x)
        x = self.pool(torch.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

# Define evaluation function
def eval_model(model, loss_fn, dataloader, use_cuda):
    """
    Evaluate the performance of a neural network model on a given dataset, 
    taking into account possible random switches of positive and negative faces.

    This function computes the average loss and adjusted accuracy of the model 
    over all batches in the provided dataloader. The accuracy calculation is specifically 
    tailored to account for cases where the positive and negative faces might have been switched. 
    It ensures the model is set to evaluation mode during evaluation to deactivate dropout 
    and batch normalization effects, and restores the original training state afterwards.

    Parameters:
        model (torch.nn.Module): The model to evaluate.
        loss_fn (callable): The loss function to use for evaluation, which should accept
          two parameters: the predicted values and the true values.
        dataloader (torch.utils.data.DataLoader): A DataLoader object providing batches
          of data (voice embeddings, two sets of face embeddings, and labels indicating if a switch occurred).
        use_cuda (bool): Flag indicating whether CUDA is enabled and available. If True, data
          will be moved to GPU for faster processing.

    Returns:
        tuple: A tuple containing the average loss and accuracy across all batches in the dataloader.
          Accuracy is computed as the mean of correct predictions, adjusted for any face switches.

    The function iterates over all batches in the dataloader, computes predictions using the model,
    and evaluates these predictions against the true labels using the specified loss function. It checks
    if the model correctly identifies the face that matches the voice, considering the possibility of 
    face switches. Results are accumulated to compute the average loss and adjusted accuracy, which are returned as a tuple.
    """
    
    is_train = model.training
    if is_train:
        model.eval()
    with torch.no_grad():
        preds_lbls = [(model(x.cuda(), f1.cuda(), f2.cuda()), y.cuda()) if use_cuda else (model(x, f1, f2), y) for x, f1, f2, y in dataloader]
        loss = np.mean([loss_fn(pred, y).item() for pred, y in preds_lbls])
        acc = np.concatenate([((pred.detach().cpu().numpy() >= 0.) == y.cpu().numpy())[:, 0] for pred, y in preds_lbls]).mean()
    if is_train:
        model.train()
    return loss, acc

# Function to plot training stats
def plot_training_stats(stats, results_folder):
    plt.figure(figsize=(12, 8))
    
    # Plot training and validation loss
    plt.subplot(2, 1, 1)
    plt.plot(stats['train_loss'], label='Training Loss')
    plt.plot(stats['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Plot validation accuracy
    plt.subplot(2, 1, 2)
    plt.plot(stats['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Validation Accuracy: {stats["val_acc"][-1]:.3f}')
    plt.legend()

    plt.tight_layout()
    plot_file = os.path.join(results_folder, 'training_stats.png')
    plt.savefig(plot_file)
    print(f'Training plot saved to {plot_file}')

# Main function adapted for Colab
def train_and_evaluate(v_embeds, f_embeds, triplets_train, triplets_val, triplets_test, results_folder, num_epochs):
    """
    Train and evaluate a Voice-Face matching model using triplets of embeddings.

    Parameters:
        v_embeds (np.ndarray): Array of voice embeddings.
        f_embeds (np.ndarray): Array of face embeddings.
        triplets_train, triplets_val, triplets_test (List[Tuple[int]]): Lists of triplet indices for training, validation, and testing.
        results_folder (str): Directory path to save model outputs and results.
        num_epochs (int): Number of epochs to train the model.

    This function initializes a model, trains it on the training set, evaluates on the validation set, 
    and tests on the test set. It handles data loading, model instantiation, training loop, 
    evaluation, and saving the model states.
    """
    print(f'Saving results to folder: {results_folder}')
    if not os.path.isdir(results_folder):
        os.makedirs(results_folder)

    train_over_pairs = False
    batch_sz = 64
    learning_rate = 0.001
    cfg = CFG
    random_switch_faces = not train_over_pairs

    training_data = VoiceFaceDataset(v_embeds, f_embeds, triplets_train, random_switch_faces)
    validation_data = VoiceFaceDataset(v_embeds, f_embeds, triplets_val, random_switch_faces)
    test_data = VoiceFaceDataset(v_embeds, f_embeds, triplets_test, random_switch_faces)

    train_dataloader = DataLoader(training_data, batch_size=batch_sz, shuffle=True)
    val_dataloader = DataLoader(validation_data, batch_size=batch_sz, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=batch_sz, shuffle=False)

    x_voice, x_face1, x_face2, train_labels = next(iter(train_dataloader))
    dims = [x_voice.shape[1], x_face1.shape[1]]
    print(f"Feature batch shape: voice {x_voice.size()}, face {x_face1.size()}")
    print(f"Labels batch shape: {train_labels.size()}")

    model = VoiceFaceTripletsClassifier(dims[0], dims[1], cfg)
    loss_fn = cfg['loss_fn']
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    sched = ExponentialLR(optimizer, gamma=0.995)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print('Using GPU')
        model.cuda()

    save_models = True
    model.train()
    stats = dict(train_loss=[], val_loss=[], val_acc=[])
    best_val_loss = np.inf
    last_saved = ''

    for epoch in range(num_epochs):
        running_loss = 0.0
        t0 = time.time()
        for i, data in enumerate(train_dataloader, 0):
            x_voice, x_face1, x_face2, labels = data
            if use_cuda:
                x_voice = x_voice.cuda()
                x_face1 = x_face1.cuda()
                x_face2 = x_face2.cuda()
                labels = labels.cuda()
            optimizer.zero_grad()
            outputs = model(x_voice, x_face1, x_face2)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        t1 = time.time()
        val_loss, val_acc = eval_model(model, loss_fn, val_dataloader, use_cuda)
        running_loss /= i
        stats['train_loss'].append(running_loss)
        stats['val_loss'].append(val_loss)
        stats['val_acc'].append(val_acc)
        is_best_str = ''
        if val_loss < best_val_loss and epoch > 0:
            best_val_loss = val_loss
            is_best_str = '(*best validation loss*)'
            if save_models:
                best_model_fnm = f'{results_folder}/best_model_epoch{epoch}.pth'
                torch.save(model.state_dict(), best_model_fnm)
                if os.path.exists(last_saved):
                    os.remove(last_saved)
                last_saved = best_model_fnm
        print(f'[epoch {epoch}, {t1 - t0:.2f} sec] loss: train {running_loss:.3f} val {val_loss:.3f} accuracy: val {val_acc:.3f} lr: {sched.get_last_lr()[0]:.2e} {is_best_str}')
        sched.step()

    if save_models:
        final_model_fnm = f'{results_folder}/final_model_epoch{epoch}.pth'
        torch.save(model.state_dict(), final_model_fnm)
        print(f'Best validation-loss model saved to {best_model_fnm}')
        print(f'Final model saved to {final_model_fnm}')

    best_fnm = best_model_fnm
    final_fnm = final_model_fnm

    final_model = VoiceFaceTripletsClassifier(dims[0], dims[1], cfg)
    final_model.load_state_dict(torch.load(final_fnm))
    if use_cuda:
        final_model.cuda()

    best_model = VoiceFaceTripletsClassifier(dims[0], dims[1], cfg)
    best_model.load_state_dict(torch.load(best_fnm))
    if use_cuda:
        best_model.cuda()

    loss, acc = eval_model(final_model, loss_fn, val_dataloader, use_cuda)
    print(f'Final model (validation set): loss {loss:.3f} acc {acc:.3f}')

    loss_best, acc_best = eval_model(best_model, loss_fn, val_dataloader, use_cuda)
    print(f'Best model  (validation set): loss {loss_best:.3f} acc {acc_best:.3f}')

    test_loss_best, test_acc_best = eval_model(best_model, loss_fn, test_dataloader, use_cuda)
    print(f'Best model  (test set):       loss {test_loss_best:.3f} acc {test_acc_best:.3f}')

    # Plot training statistics
    plot_training_stats(stats, 'VFMR_results')

    return best_fnm, final_fnm, stats

def count_parameters(model):
    """
    Count the number of learnable parameters in a PyTorch model.

    Parameters:
        model (nn.Module): The model whose parameters are to be counted.

    Returns:
        int: Total number of learnable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Execute training and evaluation
    data_file = 'triplet_data/triplets_1_negatives.pickle'
    with open(data_file, 'rb') as file:
        data = pickle.load(file)

    v_embeds, v_nms, f_embeds, f_nms, triplets, triplets_train, triplets_val, triplets_test = data

    results_folder = 'trained_VFTC_model'
    num_epochs = 120
    best_fnm, final_fnm, stats = train_and_evaluate(v_embeds, f_embeds, triplets_train, triplets_val, triplets_test, results_folder, num_epochs)

    model = VoiceFaceTripletsClassifier(input_sz_voice=192, input_sz_face=512, cfg={
    'input_layer_size': 256, 
    'dropout': 0.5
    })
    total_params = count_parameters(model)
    print(f"Total learnable parameters: {total_params}")
