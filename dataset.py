import numpy as np
import torch
from torch.utils.data import Dataset

class VoiceFaceDataset(Dataset):
    """
    A dataset class for loading and transforming voice and face embeddings into training triplets for a machine learning model.

    The class handles triplet formation, where each triplet consists of a positive voice-face pair and a negative face sample. Optionally, the class can randomly switch the positive and negative faces in the triplet to augment the dataset and prevent the model from learning trivial solutions.

    Attributes:
        v_embeds (dict): Dictionary containing voice embeddings.
        f_embeds (dict): Dictionary containing face embeddings.
        triplets (np.ndarray): Array of indices forming triplets.
        labels (np.ndarray): Array of labels indicating if faces in triplets are switched (1 for original, 0 for switched).
        v_index_to_key (dict): Mapping from index to key for voice embeddings.
        f_index_to_key (dict): Mapping from index to key for face embeddings.

    Parameters:
        v_embeds (dict): Voice embeddings as a dictionary.
        f_embeds (dict): Face embeddings as a dictionary.
        triplets (List[Tuple[int, int, int]]): A list of tuples, each containing indices for (voice, positive face, negative face).
        random_switch_faces (bool): If True, randomly switches the positive and negative faces in half of the triplets.
        random_seed (int): Seed for random operations to ensure reproducibility.
    """
    def __init__(self, v_embeds, f_embeds, triplets, v_index_to_key, f_index_to_key, random_switch_faces=False, random_seed=42):
        self.v_embeds = v_embeds
        self.f_embeds = f_embeds
        self.v_index_to_key = v_index_to_key
        self.f_index_to_key = f_index_to_key
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
        triplet = self.triplets[idx]
        vpos, f1, f2 = triplet

        # Map indices to keys
        v_key = self.v_index_to_key[vpos]
        f1_key = self.f_index_to_key[f1]
        f2_key = self.f_index_to_key[f2]

        if v_key not in self.v_embeds:
            raise KeyError(f"Key {v_key} not found in voice embeddings. Available keys: {list(self.v_embeds.keys())}")
        if f1_key not in self.f_embeds:
            raise KeyError(f"Key {f1_key} not found in face embeddings. Available keys: {list(self.f_embeds.keys())}")
        if f2_key not in self.f_embeds:
            raise KeyError(f"Key {f2_key} not found in face embeddings. Available keys: {list(self.f_embeds.keys())}")

        v = np.array(self.v_embeds[v_key], dtype=np.float32)
        f1 = np.array(self.f_embeds[f1_key], dtype=np.float32)
        f2 = np.array(self.f_embeds[f2_key], dtype=np.float32)
        labels = self.labels[idx]
        return v, f1, f2, labels
