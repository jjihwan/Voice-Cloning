import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment

def change_key(input_file, output_file, semitones):
    # Load the audio file
    audio = AudioSegment.from_file(input_file)
    samples = np.array(audio.get_array_of_samples())

    # Handle stereo and mono cases
    if audio.channels == 2:
        samples = samples.reshape((-1, 2)).T
    else:
        samples = samples.reshape((1, -1))

    # Convert the sample array to floating-point values
    samples = samples.astype(np.float32) / 32768.0

    # Perform pitch shifting
    sr = audio.frame_rate
    pitch_shifted_samples = librosa.effects.pitch_shift(samples, sr=sr, n_steps=semitones)

    # Convert the floating-point samples back to int16
    pitch_shifted_samples = (pitch_shifted_samples.T * 32768).astype(np.int16)

    # Flatten the array if it was mono
    if audio.channels == 1:
        pitch_shifted_samples = pitch_shifted_samples.flatten()

    # Create a new audio segment from the pitch-shifted samples
    new_audio = AudioSegment(
        pitch_shifted_samples.tobytes(), 
        frame_rate=sr,
        sample_width=pitch_shifted_samples.dtype.itemsize, 
        channels=audio.channels
    )

    # Export the transposed audio to a file
    new_audio.export(output_file, format="wav")

# # Example usage
# input_file = 'instrument/loveescape_Instruments.wav'  # Path to your input audio file
# output_file = 'output_transposed.wav'  # Path to save the transposed audio file
# semitones = 6  # Number of semitones to transpose (positive or negative)


# change_key(input_file, output_file, semitones)

