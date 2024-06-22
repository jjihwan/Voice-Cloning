from infer.modules.vc.modules import VC
from configs.config import Config
import torchaudio
import os
from scipy.io import wavfile
from glob import glob
from argparse import ArgumentParser

os.environ["weight_root"] = "assets/weights"
os.environ["uvr5_weights"] = "assets/uvr5_weights"
os.environ["index_root"] = "logs"
os.environ["rmvpe_root"] = "assets/rmvpe"

def main(model_name, target_dir):
    sources = [model_name]
    targets = sorted(glob(f"{target_dir}/*.wav"))
    print(targets)
    keys = [0] # change this to [0, -6, 6] to generate 3 keys
    
    for source in sources:
        cfg = Config()
        cfg.device = "cuda"
        print(cfg.device)
        vc = VC(cfg)
        vc.get_vc(f"{source}.pth")
        for target in targets:
            for key in keys:
                target_music = os.path.basename(target).split("_Vocals")[0]

                info, output = vc.vc_single(
                    sid=0,
                    input_audio_path=target,
                    f0_up_key=key,
                    f0_file=None,
                    f0_method="rmvpe",
                    file_index=f"logs/{source}/added_IVF202_Flat_nprobe_1_{source}_v2.index",
                    file_index2=None,
                    index_rate=0.66,
                    filter_radius=3,
                    resample_sr=0,
                    rms_mix_rate=0.21,
                    protect=0.33
                )

                print(info)

                sr, wav = output
                print(sr)
                print(wav.shape)

                os.makedirs(f"./results/{source}", exist_ok=True)
                wavfile.write(f"./results/{source}/{target_music}_{source}_{key}.wav", sr, wav)
                print(f"./results/{source}/{target_music}_{source}_{key}.wav")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="iu")
    parser.add_argument("--target_dir", type=str, default="target_dataset/vocals")
    args = parser.parse_args()

    main(args.model_name, args.target_dir)