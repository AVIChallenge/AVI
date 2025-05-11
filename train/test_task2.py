import os
import torch
import argparse
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_squared_error
from train import MultimodalMLP, DeepEnsembleMLP

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MultimodalDataset(Dataset):
    def __init__(self, csv_file, audio_dir, video_dir, text_dir, question, label_col, rating_csv):
        self.data = pd.read_csv(csv_file)
        self.audio_dir = audio_dir
        self.video_dir = video_dir
        self.text_dir = text_dir
        self.question = question
        self.label_col = label_col
        self.rating = pd.read_csv(rating_csv)

        self.result_dict = {}
        for _, row in self.rating.iterrows():
            key = row["id"]
            self.result_dict[key] = row

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample_id = self.data.iloc[idx]['id']
        audio_files, video_files, text_files = [], [], []

        for q in self.question:
            audio_file = [f for f in os.listdir(self.audio_dir) if f.startswith(f"{sample_id}_{q}")]
            video_file = [f for f in os.listdir(self.video_dir) if f.startswith(f"{sample_id}_{q}")]
            text_file = [f for f in os.listdir(self.text_dir) if f.startswith(f"{sample_id}_{q}")]

            if not audio_file or not video_file or not text_file:
                raise FileNotFoundError(f"Files for {sample_id}_{q} not found.")
            audio_files.append(audio_file[0])
            video_files.append(video_file[0])
            text_files.append(text_file[0])

        audio_features = np.concatenate([np.load(os.path.join(self.audio_dir, f)) for f in audio_files], axis=0)
        video_features = np.tile(
            np.concatenate([np.expand_dims(np.load(os.path.join(self.video_dir, f)), axis=0) for f in video_files], axis=0),
            (5, 1)
        )
        text_features = np.tile(
            np.concatenate([np.expand_dims(np.load(os.path.join(self.text_dir, f)), axis=0) for f in text_files], axis=0),
            (5, 1)
        )

        features = np.concatenate([audio_features, video_features, text_features], axis=-1)
        labels = np.array([(self.result_dict[sample_id][col] - 1) / 4 for col in self.label_col])

        if np.isnan(labels).any():
            raise ValueError(f"Labels for {sample_id} contain NaN values.")

        return torch.tensor(features, dtype=torch.float32), sample_id

def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_and_save(model, test_loader, output_csv, label_col):
    predictions, sample_ids = [], []
    with torch.no_grad():
        for features, ids in test_loader:
            features = features.to(device)
            outputs = model(features)
            predictions.extend((outputs.cpu().numpy() * 4 + 1))
            sample_ids.extend(ids)

    predictions = np.array(predictions)
    results = pd.DataFrame(data=predictions, columns=label_col)
    results.insert(0, 'id', sample_ids)
    results.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_dir', type=str, required=True)
    parser.add_argument('--video_dir', type=str, required=True)
    parser.add_argument('--text_dir', type=str, required=True)
    parser.add_argument('--test_csv', type=str, required=True)
    parser.add_argument('--rating_csv', type=str, required=True)
    parser.add_argument('--question', nargs='+', required=True)
    parser.add_argument('--label_col', nargs='+', required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--output_csv', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)

    args = parser.parse_args()

    input_dim = 384 + 512 + 768
    dataset = MultimodalDataset(
        args.test_csv, args.audio_dir, args.video_dir,
        args.text_dir, args.question, args.label_col, args.rating_csv
    )
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = DeepEnsembleMLP(input_dim * 30, len(args.label_col), num_mlps=32).to(device)
    model = load_model(model, args.model_path, device)

    predict_and_save(model, test_loader, args.output_csv, args.label_col)

if __name__ == '__main__':
    main()
