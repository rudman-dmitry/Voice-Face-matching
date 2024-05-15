import torch
from torch.utils.data import DataLoader
import pickle
from dataset import VoiceFaceDataset
from train_VFTC_model import VoiceFaceTripletsClassifier, eval_model

def load_data(file_path):
    """
    Load data from a pickle file.
    """
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def load_model(model_path, input_sz_voice, input_sz_face, cfg):
    """
    Load a trained model from disk.
    """
    model = VoiceFaceTripletsClassifier(input_sz_voice, input_sz_face, cfg)
    model.load_state_dict(torch.load(model_path))
    return model

def main():
    # Configuration
    model_path = 'trained_VFTC_model/final_model_epoch119.pth'
    data_path = 'triplet_data/triplets_1_negatives.pickle'
    cfg = dict(input_layer_size=256, dropout=0.5, loss_fn=torch.nn.BCEWithLogitsLoss())

    # Load data
    data = load_data(data_path)
    v_embeds, _, f_embeds, _, _, _, _, triplets_test = data

    # Prepare test dataset and dataloader
    test_data = VoiceFaceDataset(v_embeds, f_embeds, triplets_test, random_switch_faces=False)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)

    # Load model
    model = load_model(model_path, v_embeds.shape[1], f_embeds.shape[1], cfg)
    if torch.cuda.is_available():
        model.cuda()
        print('Using GPU for evaluation.')
    else:
        print('Using CPU for evaluation.')

    # Evaluate model
    use_cuda = torch.cuda.is_available()
    test_loss, test_accuracy = eval_model(model, cfg['loss_fn'], test_dataloader, use_cuda)

    print(f"Test Loss: {test_loss:.3f}")
    print(f"Test Accuracy: {test_accuracy:.3f}")

if __name__ == "__main__":
    main()
