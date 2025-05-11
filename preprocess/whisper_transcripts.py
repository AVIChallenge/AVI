import os
import whisper

class BatchTranscriber:
    def __init__(self, model_path: str, audio_folder: str, output_folder: str):
        """
        Initialize the transcriber.

        Args:
            model_path (str): Path to the Whisper model file (e.g., "./tiny.pt")
            audio_folder (str): Directory containing .wav audio files
            output_folder (str): Directory to save transcript .txt files
        """
        self.model = whisper.load_model(model_path)
        self.audio_folder = audio_folder
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)

    def transcribe_all(self):
        """
        Transcribe all .wav files in the audio folder and save their transcripts as .txt.
        """
        for audio_file in os.listdir(self.audio_folder):
            if not audio_file.lower().endswith(".wav"):
                continue
            audio_path = os.path.join(self.audio_folder, audio_file)
            result = self.model.transcribe(audio_path)
            output_path = os.path.join(self.output_folder, audio_file.replace(".wav", ".txt"))
            with open(output_path, "w") as f:
                f.write(result['text'])
            print(f"Processed {audio_path} and saved transcript to {output_path}")


if __name__ == "__main__":
    # Example usage (update these paths)
    transcriber = BatchTranscriber(
        model_path="./tiny.pt",
        audio_folder="/path/to/audio",
        output_folder="/path/to/output"
    )
    transcriber.transcribe_all()
