import os
import numpy as np
import torch
from transformers import RobertaTokenizer, RobertaModel
import warnings
warnings.filterwarnings("ignore")


class RobertaFeature:
    def __init__(self, model_name='roberta-base'):
        """
        Initialize tokenizer and model.
        """
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaModel.from_pretrained(model_name)

    def get_feature_global_pooling(self, text, max_length=512):
        """
        Get sentence embedding using pooler output with sliding window tokenization.

        Args:
            text (str): Input text.
            max_length (int): Maximum token length for each segment.

        Returns:
            torch.Tensor: Aggregated feature vector.
        """
        tokenized = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=max_length,
            stride=256,
            return_overflowing_tokens=True
        )
        pooler_outputs = []
        for encoded_input in tokenized['input_ids']:
            output = self.model(input_ids=encoded_input.unsqueeze(0))
            pooler_outputs.append(output.pooler_output.squeeze(0))

        aggregated_feature = torch.stack(pooler_outputs, dim=0).mean(dim=0)
        return aggregated_feature

    def save_feature(self, text, save_path):
        """
        Save the pooled feature as .npy file.

        Args:
            text (str): Input text.
            save_path (str): Path to save the numpy file.
        """
        feature = self.get_feature_global_pooling(text).detach().numpy()
        np.save(save_path, feature)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--text_folder", type=str, required=True, help="Directory containing .txt transcripts")
    parser.add_argument("--save_folder", type=str, required=True, help="Directory to save output features (.npy)")
    args = parser.parse_args()

    os.makedirs(args.save_folder, exist_ok=True)
    extractor = RobertaFeature()

    for filename in os.listdir(args.text_folder):
        if not filename.endswith(".txt"):
            continue

        text_path = os.path.join(args.text_folder, filename)
        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read()

        save_path = os.path.join(args.save_folder, filename.replace(".txt", ".npy"))
        extractor.save_feature(text, save_path)
        print(f"Saved feature for {filename}")
