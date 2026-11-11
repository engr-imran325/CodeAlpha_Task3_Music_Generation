AI Music Generation Using LSTM

Project Overview

This project is an AI-based music generation system that uses a Long Short-Term Memory (LSTM) neural network to learn musical patterns from classical MIDI files and generate new instrumental music.

Internship

CodeAlpha – Artificial Intelligence Internship

Technologies Used

- Python
- TensorFlow and Keras
- Music21
- NumPy
- Pretty MIDI
- SoundFile

Dataset

The model was trained using five classical MIDI files:

- Bach
- Canon in D
- Chopin
- Fur Elise
- Moonlight Sonata

Project Features

- Collects classical MIDI music data.
- Extracts notes and chords using Music21.
- Preprocesses musical sequences for training.
- Builds an LSTM deep learning model.
- Trains the model for 15 epochs.
- Generates new musical sequences.
- Converts generated MIDI music into WAV audio.

Model Architecture

The LSTM model contains:

- Embedding Layer
- First LSTM Layer (128 units)
- Dropout Layer (0.2)
- Second LSTM Layer (128 units)
- Dropout Layer (0.2)
- Dense Output Layer with Softmax Activation

Project Workflow

1. MIDI dataset collection.
2. Musical note extraction.
3. Data preprocessing and sequence preparation.
4. LSTM model creation.
5. Model training.
6. New music generation.
7. MIDI and WAV audio export.

Generated Output Files

- "music_lstm_model.keras" – Trained AI model
- "generated_music.mid" – Generated MIDI music
- "generated_music.wav" – Generated audio
- "notes.txt" – Extracted musical notes
- "note_mapping.json" – Note mapping information

Installation

Install the required libraries:

pip install -r requirements.txt

How to Run

Activate the virtual environment:

& D:\PythonEnv\MusicAI\Scripts\Activate.ps1

Run the main Python file:

python main.py

Note: Running "main.py" performs preprocessing, model creation, training, and music generation again.

To listen to the generated music, open "generated_music.wav".

Author
Muhammad Imran Shah

Project

CodeAlpha AI Internship – Task 3: Music Generation with AI
