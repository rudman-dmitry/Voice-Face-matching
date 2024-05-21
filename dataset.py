# voice_face_dataset.py

import numpy as np
from torch.utils.data import Dataset

class VoiceFaceDataset(Dataset):
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
