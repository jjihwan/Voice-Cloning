# Retrieval-based Voice Conversion

## 1. Set up ⚙️
```
conda create -n rvc python=3.10.14
conda activate rvc
pip3 install -r requirements.txt
```

## 2. Usage 🚀
First, clone our repository.
```
git clone git@github.com:jjihwan/Voice-Cloning.git
```

### 2.1. Dataset Preparation 🦄
Set model_name and make `dataset/model_name` directory. e.g. `dataset/iu`
```
mkdir -p dataset/iu # change iu as your own model name
```

1. Download source musics in `musics` folder (more than 4 songs recommended). Store them as following structure:
    ```
    musics
    └── iu
        └── boo.mp3 # names are not important
        └── foo.mp3
        └── moo.mp3
        ...
    ```
2. Download pretrained vocal-remover from the original repository
    Download pretrained model
    ```
    wget https://github.com/tsurumeso/vocal-remover/releases/download/v5.1.0/vocal-remover-v5.1.0.zip
    unzip vocal-remover-v5.1.0.zip
    mv vocal-remover/models .
    ```
3. Use vocal-remover to decompose the vocal and instrument.
    ```
    python3 vocal_remover.py --input_dir musics/iu --output_dir dataset/iu --gpu 0
    ```

### 2.2. Training 🔥
Train your own model
```
python3 RVC_train.py --model_name iu --save_frequency 50 --epochs 200
```

### 2.3. Inference 🎵
1. Prepare the target musics in the `target_musics` folder.
    ```
    target_musics
    └── boo.mp3 # names are not important
    └── foo.mp3
    └── moo.mp3
        ...
    ```
2. Run vocal-remover
    ```
    python3 vocal_remover.py --input_dir target_musics --output_dir target_dataset --gpu 0
    ```

3. Inference with your own model
    ```
    python3 RVC_inference.py --model_name iu --target_dir target_dataset/vocals
    ```
    You might have to modify L17 to adjust the key if the model and the target keys are different.
    You can find the results in the `results` folder.

4. Compose with instruments
    ```
    python3 compose_song.py --model_name iu --target_dir target_dataset/instruments
    ```

### 2.4. One-click Training & Inference 🤩

Run the `run.sh` file after preparing the training datasets(in 2.1.1) and target musics(in 2.3.1).
You can train the model and inference by it at once! 🔥🔥🔥
```
sh run.sh iu
```


## Acknowledgement 🤗🤗🤗
Our codes are built on two nice open-source projects, [RVC-project](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git) and [Vocal-Remover](https://github.com/tsurumeso/vocal-remover.git). Thanks for the authors!