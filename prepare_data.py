import os
import pickle
import itertools
import numpy as np
import pandas as pd
import difflib
from typing import List, Tuple, Dict

def clean_string(name: str) -> str:
    """
    Remove non-ASCII characters from a string.

    Args:
        name (str): The input string.

    Returns:
        str: A cleaned string with only ASCII characters.
    """
    return ''.join(char for char in name if ord(char) < 128)

def split_by_name(names: np.ndarray, ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1), seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split names into training, validation, and test sets.

    Args:
        names (np.ndarray): Array of unique names.
        ratios (Tuple[float, float, float]): Proportions for train, validation, and test sets.
        seed (int): Random seed for reproducibility.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Training, validation, and test names.
    """
    np.random.seed(seed)
    np.random.shuffle(names)
    train_size = int(ratios[0] * len(names))
    val_size = int(ratios[1] * len(names))
    train_names = names[:train_size]
    val_names = names[train_size:train_size + val_size]
    test_names = names[train_size + val_size:]
    return train_names, val_names, test_names

def resolve_name_mismatches(audio_names, image_names):
    """
    Attempt to find the best match for names that do not directly match between
    audio and image datasets using string similarity.

    Args:
        audio_names (set): Unique names from the audio dataset.
        image_names (set): Unique names from the image dataset.

    Returns:
        dict: A dictionary where keys are audio names and values are the best matching image names.
    """
    matched_names = {}
    # Find unmatched names in audio not present in images
    for name in audio_names:
        # Get the closest match from image names
        close_matches = difflib.get_close_matches(name, image_names, n=1, cutoff=0.6)
        if close_matches:
            matched_names[name] = close_matches[0]
        else:
            matched_names[name] = None  # No close match found

    return matched_names

def reconcile_names(audio_df, image_df, matches):
    """
    Reconcile names between audio and image datasets based on provided matches.
    Args:
        audio_df (pd.DataFrame): DataFrame with audio data.
        image_df (pd.DataFrame): DataFrame with image data.
        matches (dict): Dictionary with audio names as keys and matched image names as values.
    Returns:
        pd.DataFrame, pd.DataFrame: Updated DataFrames with reconciled names.
    """
    # Update the names in the audio DataFrame
    audio_df['name'] = audio_df['name'].apply(lambda x: matches[x] if x in matches and matches[x] else x)
    # Ensure both DataFrames use the same set of names
    common_names = set(audio_df['name']).intersection(set(image_df['name']))
    audio_df = audio_df[audio_df['name'].isin(common_names)]
    image_df = image_df[image_df['name'].isin(common_names)]
    return audio_df, image_df

def create_triplet_samples(audio_df: pd.DataFrame, image_df: pd.DataFrame, negatives_per_positive: int, seed: int = 42) -> List[Tuple[int, int, int]]:
    """
    Create triplet samples of positive and negative pairs.

    Args:
        audio_df (pd.DataFrame): DataFrame containing audio names and their splits.
        image_df (pd.DataFrame): DataFrame containing image names and their splits.
        negatives_per_positive (int): Number of negative samples per positive pair.
        seed (int): Random seed for reproducibility.

    Returns:
        List[Tuple[int, int, int]]: List of triplet indices.
    """
    np.random.seed(seed)
    triplets = []

    for name in audio_df['name'].unique():
        audio_positives = audio_df[audio_df['name'] == name]
        image_positives = image_df[image_df['name'] == name]
        positive_pairs = list(itertools.product(audio_positives.index, image_positives.index))
        split = audio_positives['split'].iloc[0]

        image_negatives = image_df[(image_df['name'] != name) & (image_df['split'] == split)]
        negative_samples_needed = len(positive_pairs) * negatives_per_positive
        negative_indices = image_negatives.sample(negative_samples_needed, replace=True).index

        negative_indices = negative_indices.values.reshape(len(positive_pairs), negatives_per_positive)

        for pair, negs in zip(positive_pairs, negative_indices):
            triplets.extend((pair[0], pair[1], neg) for neg in negs)

    return triplets

def main() -> None:
    """
    Main function to load embeddings, generate triplets, and save the dataset.
    """
    # Load image and audio embeddings from pickle files
    with open('vfm_assignment/image_embeddings.pickle', 'rb') as img_file:
        image_embeddings: Dict[str, np.ndarray] = pickle.load(img_file)

    with open('vfm_assignment/audio_embeddings.pickle', 'rb') as audio_file:
        audio_embeddings: Dict[str, np.ndarray] = pickle.load(audio_file)

    # Extract and clean names from embeddings keys
    image_keys = list(image_embeddings.keys())
    audio_keys = list(audio_embeddings.keys())

    image_names = [clean_string(key.split('/')[0]) for key in image_keys]
    audio_names = [clean_string(key.split('/')[0]) for key in audio_keys]

    # Create DataFrames
    image_df = pd.DataFrame({'name': image_names, 'embeddings': list(image_embeddings.values())})
    audio_df = pd.DataFrame({'name': audio_names, 'embeddings': list(audio_embeddings.values())})

    # Resolve name mismatches using the closest match strategy
    matches = resolve_name_mismatches(set(audio_names), set(image_names))
    audio_df, image_df = reconcile_names(audio_df, image_df, matches)

    # Split names into training, validation, and test sets
    unique_names = np.unique(audio_df['name'])  # Ensure unique names from reconciled names
    train_names, val_names, test_names = split_by_name(unique_names)

    # Assign splits based on the split names
    audio_df['split'] = audio_df['name'].apply(lambda name: 'train' if name in train_names else ('val' if name in val_names else 'test'))
    image_df['split'] = image_df['name'].apply(lambda name: 'train' if name in train_names else ('val' if name in val_names else 'test'))

        # Save the DataFrames with splits
    audio_df.to_pickle('triplet_data/audio_df_with_splits.pickle')
    image_df.to_pickle('triplet_data/image_df_with_splits.pickle')

    # Generate triplets and save them
    output_dir = 'triplet_data'
    os.makedirs(output_dir, exist_ok=True)

    for negatives_per_positive in [1, 4]:
        print(f'Creating triplets with {negatives_per_positive} negatives per positive pair...')
        triplets = create_triplet_samples(audio_df, image_df, negatives_per_positive)
        triplets_array = np.array(triplets)
        triplets_train = triplets_array[np.isin(triplets_array[:, 0], audio_df[audio_df['split'] == 'train'].index)]
        triplets_val = triplets_array[np.isin(triplets_array[:, 0], audio_df[audio_df['split'] == 'val'].index)]
        triplets_test = triplets_array[np.isin(triplets_array[:, 0], audio_df[audio_df['split'] == 'test'].index)]

        output_file = f'{output_dir}/triplets_{negatives_per_positive}_negatives.pickle'
        with open(output_file, 'wb') as file:
            # Save the triplets and embeddings to a pickle file
            pickle.dump([audio_embeddings, audio_df, image_embeddings, image_df, triplets_array, triplets_train, triplets_val, triplets_test], file)
        
        print(f'Saved triplets to {output_file}')

if __name__ == "__main__":
    main()

