from pydub import AudioSegment
import wave
from glob import glob
import os

from change_key import change_key
from argparse import ArgumentParser

def combine(voice_file, inst_file, combine_file):
  sound1 = AudioSegment.from_file(voice_file)
  sound2 = AudioSegment.from_file(inst_file)
  combined = sound1.overlay(sound2)
  combined.export(combine_file, format='wav')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="iu")
    parser.add_argument("--target_dir", type=str, default="target_dataset/instruments")

    args = parser.parse_args()

    voice_files = sorted(glob(f"results/{args.model_name}/*.wav"))

    for voice_file in voice_files:
        key = int(voice_file.split("/")[-1].split("_")[-1].split(".")[0])
        output_path = voice_file.replace("results", "results_combined")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        music_name = voice_file.split("/")[-1].split(f"_{args.model_name}")[0]
        print(music_name)
        inst_file = f"{args.target_dir}/{music_name}_Instruments.wav"
        if key == 0:
          inst_file_transposed = inst_file
        else:
          inst_file_transposed = f"{args.target_dir}/{music_name}_Instruments_{key}.wav"

        if not os.path.exists(inst_file_transposed):
          change_key(inst_file, inst_file_transposed, key)
        combine(voice_file, inst_file_transposed, output_path)
