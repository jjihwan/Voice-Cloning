model_name="$1"

python3 RVC_train.py --model_name $model_name --save_frequency 50 --epochs 200
python3 vocal_remover.py --input_dir target_musics --output_dir target_dataset --gpu 0
python3 RVC_inference.py --model_name $model_name --target_dir target_dataset/vocals
python3 compose_song.py --model_name $model_name --target_dir target_dataset/instruments