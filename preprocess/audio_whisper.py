import os
import numpy as np
import sys
sys.path.append(".")

import whisper


class Audio2Feature():
    def __init__(self, model_path="./tiny.pt", audio_folder="/path/to/audio", feature_folder="/path/to/feature"):
        """
        :param model_path: path to Whisper model
        :param audio_folder: folder containing input audio files
        :param feature_folder: folder to store extracted features
        """
        self.model = whisper.load_model(model_path)
        self.audio_folder = audio_folder
        self.feature_folder = feature_folder

        if not os.path.exists(self.feature_folder):
            os.makedirs(self.feature_folder)

    def get_sliced_feature(self, feature_array, vid_idx, audio_feat_length=[2,2], fps=25):
        """
        Get sliced features based on a given index
        :param feature_array: full feature array
        :param vid_idx: video frame index
        :param audio_feat_length: context length [left, right]
        :param fps: frames per second of the video
        :return: sliced features and selected indices
        """
        length = len(feature_array)
        selected_feature = []
        selected_idx = []

        center_idx = int(vid_idx * 50 / fps) 
        left_idx = center_idx - audio_feat_length[0] * 2
        right_idx = center_idx + (audio_feat_length[1] + 1) * 2

        for idx in range(left_idx, right_idx):
            idx = max(0, min(length - 1, idx))
            selected_feature.append(feature_array[idx])
            selected_idx.append(idx)

        selected_feature = np.concatenate(selected_feature, axis=0)
        selected_feature = selected_feature.reshape(-1, 384)
        return selected_feature, selected_idx

    def get_sliced_feature_sparse(self, feature_array, vid_idx, audio_feat_length=[2,2], fps=25):
        """
        Get sparsely sampled sliced features based on a given index
        :param feature_array: full feature array
        :param vid_idx: video frame index
        :param audio_feat_length: context length [left, right]
        :param fps: frames per second of the video
        :return: sliced features and selected indices
        """
        length = len(feature_array)
        selected_feature = []
        selected_idx = []

        for dt in range(-audio_feat_length[0], audio_feat_length[1] + 1):
            idx = int((vid_idx + dt) * 50 / fps)
            idx = max(0, min(length - 1, idx))

            if idx < 1 or idx > length - 1:
                x = feature_array[idx][np.newaxis, :, :]
                x = np.repeat(x, 2, axis=0)
                selected_feature.append(x)
                selected_idx.extend([idx, idx])
            else:
                x = feature_array[idx - 1:idx + 1]
                selected_feature.append(x)
                selected_idx.extend([idx - 1, idx])

        selected_feature = np.concatenate(selected_feature, axis=0)
        selected_feature = selected_feature.reshape(-1, 384)
        return selected_feature, selected_idx

    def feature2chunks(self, feature_array, fps, audio_feat_length=[2,2]):
        """
        Divide feature array into chunks for each frame
        :param feature_array: full feature array
        :param fps: frames per second of the video
        :param audio_feat_length: context length [left, right]
        :return: list of feature chunks
        """
        whisper_chunks = []
        whisper_idx_multiplier = 50. / fps
        i = 0
        print(f"Video at {fps} FPS, audio indexed at 50 FPS")
        while True:
            start_idx = int(i * whisper_idx_multiplier)
            selected_feature, _ = self.get_sliced_feature(
                feature_array=feature_array,
                vid_idx=i,
                audio_feat_length=audio_feat_length,
                fps=fps
            )
            whisper_chunks.append(selected_feature)
            i += 1
            if start_idx > len(feature_array):
                break
        return whisper_chunks

    def audio2feat(self, audio_path):
        """
        Convert audio to feature array
        :param audio_path: path to the audio file
        :return: concatenated feature array
        """
        result = self.model.transcribe(audio_path)
        embed_list = []
        for emb in result['segments']:
            encoder_embeddings = emb['encoder_embeddings'].transpose(0, 2, 1, 3).squeeze(0)
            start_idx = int(emb['start'])
            end_idx = int(emb['end'])
            emb_end_idx = int((end_idx - start_idx) / 2)
            embed_list.append(encoder_embeddings[:emb_end_idx])
        return np.concatenate(embed_list, axis=0)

    def infer(self, audio_path):
        """
        Transcribe audio to text and other metadata
        :param audio_path: path to the audio file
        :return: result dictionary from whisper
        """
        return self.model.transcribe(audio_path)

    def process_all_audio(self):
        """
        Process all audio files in the specified folder and save features
        """
        audio_files = [f for f in os.listdir(self.audio_folder) if f.endswith(".wav")]
        for filename in audio_files:
            audio_path = os.path.join(self.audio_folder, filename)
            feature_path = os.path.join(self.feature_folder, filename.replace(".wav", ".npy"))
            print(f"Processing {audio_path}")
            if os.path.exists(feature_path):
                print(f"{feature_path} exists, skipping.")
                continue
            result = self.audio2feat(audio_path)
            np.save(feature_path, result)


if __name__ == "__main__":
    audio_folder = "/path/to/audio"
    feature_folder = "/path/to/whisper_feature"
    model = Audio2Feature(audio_folder=audio_folder, feature_folder=feature_folder)
    model.process_all_audio()
