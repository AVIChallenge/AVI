# 重构后的 train_task1.py

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error


class MultimodalDataset(Dataset):
    def __init__(self, csv_file, audio_dir, video_dir, text_dir, question, label_col):
        self.data = pd.read_csv(csv_file)
        self.audio_dir = audio_dir
        self.video_dir = video_dir
        self.text_dir = text_dir
        self.question = question
        self.label_col = label_col

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

        video_features = np.tile(np.expand_dims(video_features, 0), (5, 1))
        text_features = np.tile(np.expand_dims(text_features, 0), (5, 1))

        features = np.concatenate([audio_features, video_features, text_features], axis=-1)
        label = self.data.iloc[idx][self.label_col]
        label_normalized = (label - 1) / 4

        return torch.tensor(features, dtype=torch.float32), torch.tensor(label_normalized, dtype=torch.float32)


class DeepEnsembleMLP(nn.Module):
    def __init__(self, input_dim, output_dim, num_mlps=32):
        super().__init__()
        self.ensemble = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 8),
                nn.ReLU(),
                nn.Linear(8, output_dim)
            ) for _ in range(num_mlps)
        ])

    def forward(self, x):
        x = x.view(x.size(0), -1)
        outputs = torch.stack([mlp(x) for mlp in self.ensemble], dim=0)
        return outputs.mean(dim=0)


def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def evaluate_model(model, loader, criterion, device):
    model.eval()
    total_loss, predictions, targets = 0, [], []
    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs.squeeze(), labels)
            total_loss += loss.item()
            predictions.append(outputs.squeeze().cpu().numpy())
            targets.append(labels.cpu().numpy())
    return total_loss / len(loader), mean_squared_error(np.concatenate(targets), np.concatenate(predictions))


def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', required=True)
    parser.add_argument('--val_csv', required=True)
    parser.add_argument('--test_csv', required=True)
    parser.add_argument('--audio_dir', required=True)
    parser.add_argument('--video_dir', required=True)
    parser.add_argument('--text_dir', required=True)
    parser.add_argument('--question', required=True)
    parser.add_argument('--label_col', required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--output_model', default='best_model.pth')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = 384 + 512 + 768

    train_set = MultimodalDataset(args.train_csv, args.audio_dir, args.video_dir, args.text_dir, args.question, args.label_col)
    val_set = MultimodalDataset(args.val_csv, args.audio_dir, args.video_dir, args.text_dir, args.question, args.label_col)
    test_set = MultimodalDataset(args.test_csv, args.audio_dir, args.video_dir, args.text_dir, args.question, args.label_col)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    model = DeepEnsembleMLP(input_dim * 5, 1, num_mlps=32).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val_loss = float('inf')
    for epoch in range(args.num_epochs):
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        val_loss, val_mse = evaluate_model(model, val_loader, criterion, device)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, args.output_model)
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val MSE={val_mse:.4f}")

    model.load_state_dict(torch.load(args.output_model))
    model.eval()
    test_loss, test_mse = evaluate_model(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test MSE: {test_mse:.4f}")


if __name__ == '__main__':
    main()
