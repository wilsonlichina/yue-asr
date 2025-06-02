# Yue ASR (Cantonese Speech Recognition) Comparison Tool

This project compares the transcription accuracy of two Automatic Speech Recognition (ASR) systems for Cantonese (Yue) audio:
1. A custom Whisper model deployed on Amazon SageMaker
2. Amazon Transcribe service

## Project Overview

The tool processes Cantonese audio files stored in an S3 bucket, transcribes them using both ASR systems, and compares the results against reference transcriptions (labels). The results are saved to a CSV file for analysis.

## Features

- Retrieves audio files from an Amazon S3 bucket
- Transcribes audio using a custom Whisper model deployed on SageMaker
- Transcribes the same audio using Amazon Transcribe service
- Compares both transcriptions with reference labels
- Outputs results to a CSV file for analysis

## Prerequisites

- AWS account with access to:
  - Amazon S3
  - Amazon SageMaker
  - Amazon Transcribe
- Python 3.6+
- Required Python packages (see requirements.txt)
- A deployed Whisper model endpoint on SageMaker
- Audio files in WAV format stored in an S3 bucket
- Reference transcriptions in CSV format

## Setup

1. Clone this repository
2. Install required packages:
   ```
   pip install -r requirements.txt
   ```
3. Configure AWS credentials:
   ```
   aws configure
   ```
4. Update the following variables in `main.py`:
   - `endpoint_name`: Your SageMaker endpoint name
   - `s3_bucket`: Your S3 bucket containing audio files
   - `labels_file`: Path to your reference transcriptions CSV
   - `output_file`: Desired output file name

## Usage

Run the main script:

```
python main.py
```

The script will:
1. Load reference transcriptions from the labels file
2. List all WAV files in the specified S3 bucket
3. Process each audio file with both ASR systems
4. Save the results to the specified output CSV file

## Output Format

The output CSV file contains the following columns:
- `filename`: Name of the audio file
- `whisper_transcription`: Transcription from the Whisper model
- `transcribe_transcription`: Transcription from Amazon Transcribe
- `label`: Reference transcription from the labels file

## Notes

- The project is specifically designed for Cantonese (zh-HK) audio files
- Audio files should be in WAV format
- The reference labels CSV should have columns for 'id' and 'label'
