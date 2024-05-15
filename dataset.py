import numpy as np
import torch
from torch.utils.data import Dataset

# Define the Dataset class
class VoiceFaceDataset(Dataset):
    """
    A dataset class for loading and transforming voice and face embeddings into training triplets for a machine learning model.

    The class handles triplet formation, where each triplet consists of a positive voice-face pair and a negative face sample. Optionally, the class can randomly switch the positive and negative faces in the triplet to augment the dataset and prevent the model from learning trivial solutions.

    Attributes:
        v_embeds (np.ndarray): Numpy array containing voice embeddings.
        f_embeds (np.ndarray): Numpy array containing face embeddings.
        triplets (np.ndarray): Array of indices forming triplets.
        labels (np.ndarray): Array of labels indicating if faces in triplets are switched (1 for original, 0 for switched).

    Parameters:
        v_embeds (np.ndarray): Voice embeddings as a float32 numpy array.
        f_embeds (np.ndarray): Face embeddings as a float32 numpy array.
        triplets (List[Tuple[int, int, int]]): A list of tuples, each containing indices for (voice, positive face, negative face).
        random_switch_faces (bool): If True, randomly switches the positive and negative faces in half of the triplets.
        random_seed (int): Seed for random operations to ensure reproducibility.
    """
    def __init__(self, v_embeds, f_embeds, triplets, random_switch_faces=False, random_seed=42):
        self.v_embeds = v_embeds.astype(np.float32)
        self.f_embeds = f_embeds.astype(np.float32)
        self.triplets = triplets.copy()
        N = len(self.triplets)
        y = np.ones((N, 1))
        if random_switch_faces:
            np.random.seed(random_seed)
            fpos_neg = self.triplets[:, 1:]
            i_switch = np.random.choice(N, N // 2, replace=False)
            fpos_neg[i_switch] = fpos_neg[i_switch, ::-1]
            self.triplets[:, 1:] = fpos_neg
            y[i_switch] = 0.
        self.labels = y

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        """
        Retrieves a triplet at the specified index along with its label.

        Parameters:
            idx (int): The index of the triplet.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Tuple containing embeddings for the voice, positive face, negative face, and the label.
        """
        triplets = self.triplets[idx]
        vpos, f1, f2 = triplets.T
        v = self.v_embeds[vpos, :]
        f1 = self.f_embeds[f1, :]
        f2 = self.f_embeds[f2, :]
        labels = self.labels[idx]
        return v, f1, f2, labels