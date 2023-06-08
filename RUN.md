# Running step

## 1. Prepare data

### 1.1. Data Preprocessing

```bash
python process_data.py -d ./input/genea2023_dataset -w ./input/word_embedding/crawl-300d-2M.vec
```

The above script generates the following four h5 files under the project directory:

```txt
trn_interloctr_v0.h5
trn_main-agent_v0.h5
val_interloctr_v0.h5
val_main-agent_v0.h5
```

### 1.2. Calculate audio statistics

```bash
python calculate_audio_statistics.py
```

### 1.3. Create Motion Processing Pipelines

```bash
python create_pipeline.py
```

## 2. Train the Model

### 2.1 Monadic

```bash
cd Tacotron2
python train_monadic.py
```

### 2.2 Dyadic

```bash
cd Tacotron2
python train_dyadic.py
```

### Output

```txt
output_directory
```


## 3. Testing the Model

```bash
python generate_all_gestures.py -ch <checkpoint_path> -t full
```

## 4. Visualization

```bash
blender -b --python "<path to 'blender_render_2023.py' script>" -- -i1 "<path to main agent BVH file>" -i2 "<path to interlocutor BVH file>" -a1 "<path to main agent WAV file>" -a2 "<path to interlocutor WAV file>" -v -d 600 -o <directory to save MP4 video in> -m <visualization mode>
```
