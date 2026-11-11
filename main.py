import os
from music21 import converter, note, chord

DATASET_PATH = "dataset/classical"

all_notes = []

for filename in os.listdir(DATASET_PATH):
    if filename.endswith((".mid", ".midi")):
        file_path = os.path.join(DATASET_PATH, filename)

        try:
            midi = converter.parse(file_path)

            for element in midi.flatten().notes:
                if isinstance(element, note.Note):
                    all_notes.append(str(element.pitch))

                elif isinstance(element, chord.Chord):
                    all_notes.append(".".join(
                        str(pitch) for pitch in element.pitches
                    ))

            print(f"Processed: {filename}")

        except Exception as error:
            print(f"Error in {filename}: {error}")

print("\nTotal notes extracted:", len(all_notes))

with open("notes.txt", "w") as file:
    file.write("\n".join(all_notes))

print("Notes saved successfully in notes.txt!")
import numpy as np

# Convert musical notes into numbers
unique_notes = sorted(set(all_notes))

note_to_int = {
    note: number
    for number, note in enumerate(unique_notes)
}

# Create training sequences
sequence_length = 40

network_input = []
network_output = []

for i in range(len(all_notes) - sequence_length):
    sequence = all_notes[i:i + sequence_length]
    next_note = all_notes[i + sequence_length]

    network_input.append([
        note_to_int[n] for n in sequence
    ])

    network_output.append(note_to_int[next_note])

# Convert data into NumPy arrays
network_input = np.array(network_input)
network_output = np.array(network_output)

# Save prepared training data
np.save("network_input.npy", network_input)
np.save("network_output.npy", network_output)

print("\nTraining data prepared successfully!")
print("Unique notes:", len(unique_notes))
print("Input shape:", network_input.shape)
print("Output shape:", network_output.shape)

# Save note mapping
import json

with open("note_mapping.json", "w") as file:
    json.dump(note_to_int, file, indent=4)

print("Note mapping saved successfully!")
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

print("\nBuilding LSTM Music Generation Model...")

model = Sequential([
    Embedding(
        input_dim=len(unique_notes),
        output_dim=64
    ),

    LSTM(128, return_sequences=True),
    Dropout(0.2),

    LSTM(128),
    Dropout(0.2),

    Dense(len(unique_notes), activation="softmax")
])

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)
model.build(input_shape=(None, sequence_length))
model.summary()

model.save("music_lstm_model.keras")

print("\nLSTM model created and saved successfully!")
print("\nStarting LSTM Model Training...")

history = model.fit(
    network_input,
    network_output,
    epochs=15,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

model.save("music_lstm_model.keras")

print("\nModel training completed successfully!")
print("Trained model saved!")
import random
import numpy as np
from music21 import stream, note, chord

print("\nGenerating New Music...")

# Reverse note mapping
int_to_note = {
    number: musical_note
    for musical_note, number in note_to_int.items()
}

# Select a random starting sequence
start = random.randint(0, len(network_input) - 1)

pattern = list(network_input[start])
generated_notes = []

# Generate 200 new notes
for _ in range(200):

    prediction_input = np.array([pattern])

    prediction = model.predict(
        prediction_input,
        verbose=0
    )[0]

    index = np.random.choice(
        len(prediction),
        p=prediction
    )

    result = int_to_note[index]
    generated_notes.append(result)

    pattern.append(index)
    pattern = pattern[1:]

print("Music notes generated:", len(generated_notes))

# Convert notes into MIDI
music_stream = stream.Stream()

for item in generated_notes:

    if "." in item:
        pitches = item.split(".")
        music_chord = chord.Chord(pitches)
        music_chord.quarterLength = 0.5
        music_stream.append(music_chord)

    else:
        music_note = note.Note(item)
        music_note.quarterLength = 0.5
        music_stream.append(music_note)

music_stream.write(
    "midi",
    fp="generated_music.mid"
)

print("MIDI file saved successfully!")
import pretty_midi
import soundfile as sf

print("\nConverting MIDI to Audio...")

midi_data = pretty_midi.PrettyMIDI(
    "generated_music.mid"
)

audio = midi_data.synthesize(fs=22050)

sf.write(
    "generated_music.wav",
    audio,
    22050
)

print("Audio generated successfully!")
print("File: generated_music.wav")