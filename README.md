# Preprocess

## 1、Audio - Audio2Feature.py

Using OpenAI's [Whisper](https://github.com/openai/whisper) model, It processes `.wav` audio files and saves frame-aligned feature arrays, which can be used for tasks like audiovisual learning or animation generation.


### Directory Structure

```
/path/to/audio/             # Input directory for .wav audio files  
/path/to/whisper_feature/   # Output directory for extracted features (.npy)
```

### Usage

#### 1. Install Requirements

```bash
pip install git+https://github.com/openai/whisper.git
```

#### 2. Prepare Audio Files

Place your `.wav` audio files into a directory (e.g., `/path/to/audio/`).

#### 3. Run the Script

Modify the `__main__` section or run the script directly:

```python
if __name__ == "__main__":
    audio_folder = "/path/to/audio"
    feature_folder = "/path/to/whisper_feature"
    model = Audio2Feature(audio_folder=audio_folder, feature_folder=feature_folder)
    model.process_all_audio()
```

Alternatively, you can call it from another script or add CLI support using `argparse`.

#### 4. Output

For each `.wav` file, a `.npy` file will be saved in the feature folder, containing the extracted feature array.



## 2、Text - whisper_transcripts.py & text_roberta.py

This repository provides a complete pipeline for converting `.wav` audio files into sentence-level feature vectors using OpenAI's [Whisper](https://github.com/openai/whisper) and HuggingFace's [RoBERTa](https://huggingface.co/roberta-base). This process is suitable for tasks like speech-driven animation, emotion recognition, or multimodal learning.

---

### 1. Audio → Text (Whisper)

Extract transcripts from audio files using the Whisper model.

#### Directory Structure

```
/path/to/audio/             # Input directory for .wav audio files  
/path/to/transcripts/       # Output directory for Whisper-generated text (.txt)
```


#### Install Whisper

```bash
pip install git+https://github.com/openai/whisper.git
```

#### Run Transcription Script

```python
from whisper_batch import BatchTranscriber

transcriber = BatchTranscriber(
    model_path="./tiny.pt",
    audio_folder="/path/to/audio",
    output_folder="/path/to/transcripts"
)
transcriber.transcribe_all()
```

Each `.wav` file will generate a `.txt` file in the output folder.

---

### 2. Text → Feature (RoBERTa)

Convert transcribed text into sentence-level feature vectors using RoBERTa's `pooler_output`.

#### Directory Structure

```
/path/to/transcripts/       # Input directory for .txt transcript files  
/path/to/roberta_feature/   # Output directory for RoBERTa feature vectors (.npy)
```

#### Install Transformers

```bash
pip install transformers
```

#### Run Feature Extraction Script

```bash
python text2feature.py \
  --text_folder /path/to/transcripts \
  --save_folder /path/to/roberta_feature
```

Each `.txt` file will generate a `.npy` file containing the pooled RoBERTa features.


## 3、Video - Clip
#### Run Feature Extraction Script
```
python video_clip.py \
    --input_dir /path/to/videos \
    --output_dir /path/to/save/features \
    --num_frames 128 \
    --pool_type mean \
    --max_workers 6
```



# Training
### ✅ Task 1: Single-question Prediction

#### Train

```bash
python train_task1.py \
    --train_csv path/to/train.csv \
    --val_csv path/to/val.csv \
    --test_csv path/to/test.csv \
    --audio_dir path/to/audio_features \
    --video_dir path/to/video_features \
    --text_dir path/to/text_features \
    --question q6 \
    --label_col p_C_observer \
    --model_path saved_model_q6.pth \
    --batch_size 32 \
    --learning_rate 0.001 \
    --num_epochs 100
```
#### Test
```bash
python test_task1.py \
    --test_csv path/to/test.csv \
    --audio_dir path/to/audio_features \
    --video_dir path/to/video_features \
    --text_dir path/to/text_features \
    --question q6 \
    --label_col p_C_observer \
    --model_path saved_model_q6.pth \
    --output_csv predictions_task1.csv \
    --batch_size 32
```

### ✅ Task 2: Multi-question Joint Prediction
#### Train
```bash
python train_task2.py \
    --train_csv path/to/train.csv \
    --val_csv path/to/val.csv \
    --test_csv path/to/test.csv \
    --audio_dir path/to/audio_features \
    --video_dir path/to/video_features \
    --text_dir path/to/text_features \
    --rating_csv path/to/prolific_ground_truth_rating.csv \
    --question q1 q2 q3 q4 q5 q6 \
    --label_col mean_rating_Integr mean_rating_Colleg mean_rating_Soc_vers mean_rating_dev_orient mean_rating_hirea \
    --model_path saved_model_task2.pth \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --num_epochs 100
```
#### Test
```bash
python test_task2.py \
    --test_csv path/to/test.csv \
    --audio_dir path/to/audio_features \
    --video_dir path/to/video_features \
    --text_dir path/to/text_features \
    --rating_csv path/to/prolific_ground_truth_rating.csv \
    --question q1 q2 q3 q4 q5 q6 \
    --label_col mean_rating_Integr mean_rating_Colleg mean_rating_Soc_vers mean_rating_dev_orient mean_rating_hirea \
    --model_path saved_model_task2.pth \
    --output_csv predictions_task2.csv \
    --batch_size 32
```
# Eval

`/eval` folder contains a Python script to evaluate prediction results against ground truth ratings using common regression metrics.

## Script Overview

The script compares the predicted and ground truth values across multiple specified columns and computes the following evaluation metrics:

* **Mean Absolute Error (MAE)**
* **Mean Squared Error (MSE)**
* **R-squared (R2)**
* **Pearson Correlation Coefficient (R)**
* **p-value of Pearson Correlation (R\_P)**

## Usage

```bash
python evaluate.py \
    --gt_file <path_to_ground_truth_csv> \
    --pred_file <path_to_prediction_csv> \
    --cols <comma_separated_column_names> [--debug]
```

### Arguments

* `--gt_file`: Path to the ground truth CSV file
* `--pred_file`: Path to the predicted values CSV file
* `--cols`: Comma-separated names of the columns to be evaluated (e.g., `mean_rating_Integr,mean_rating_Colleg`)
* `--debug`: (Optional) If specified, the script prints individual ground truth and prediction pairs


