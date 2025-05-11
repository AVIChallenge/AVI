import os
import torch
import argparse
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
from train import MultimodalMLP, DeepEnsembleMLP

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MultimodalDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, audio_dir, video_dir, text_dir, question):
        self.data = pd.read_csv(csv_file)
        self.audio_dir = audio_dir
        self.video_dir = video_dir
        self.text_dir = text_dir
        self.question = question

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample_id = self.data.iloc[idx]['id']
        audio_file = [f for f in os.listdir(self.audio_dir) if f.startswith(f"{sample_id}_{self.question}")]
        video_file = [f for f in os.listdir(self.video_dir) if f.startswith(f"{sample_id}_{self.question}")]
        text_file = [f for f in os.listdir(self.text_dir) if f.startswith(f"{sample_id}_{self.question}")]

        if len(audio_file) == 0 or len(video_file) == 0 or len(text_file) == 0:
            raise FileNotFoundError(f"Files for {sample_id}_{self.question} not found.")

        audio_features = np.load(os.path.join(self.audio_dir, audio_file[0]))
        video_features = np.load(os.path.join(self.video_dir, video_file[0]))
        text_features = np.load(os.path.join(self.text_dir, text_file[0]))

        video_features_expanded = np.tile(np.expand_dims(video_features, axis=0), (5, 1))
        text_features_expanded = np.tile(np.expand_dims(text_features, axis=0), (5, 1))

        features = np.concatenate([audio_features, video_features_expanded, text_features_expanded], axis=-1)

        return torch.tensor(features, dtype=torch.float32), sample_id

def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_and_save(model, test_loader, output_csv, col):
    predictions = []
    sample_ids = []

    with torch.no_grad():
        for features, ids in test_loader:
            features = features.to(device)
            outputs = model(features)
            predictions.extend((outputs.squeeze().cpu().numpy() * 4 + 1))
            sample_ids.extend(ids)

    results = pd.DataFrame({'id': sample_ids, col: predictions})
    results.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_dir', type=str, required=True)
    parser.add_argument('--video_dir', type=str, required=True)
    parser.add_argument('--text_dir', type=str, required=True)
    parser.add_argument('--test_csv', type=str, required=True)
    parser.add_argument('--question', type=str, required=True)
    parser.add_argument('--label_col', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--output_csv', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)

    args = parser.parse_args()

    input_dim = 384 + 512 + 768
    test_dataset = MultimodalDataset(args.test_csv, args.audio_dir, args.video_dir, args.text_dir, args.question)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = DeepEnsembleMLP(input_dim * 5, 1, num_mlps=32).to(device)
    model = load_model(model, args.model_path, device)

    predict_and_save(model, test_loader, args.output_csv, args.label_col)

if __name__ == '__main__':
    main()
