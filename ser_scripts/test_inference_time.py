from tqdm import tqdm
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
    UniSpeechForSequenceClassification,
    UniSpeechSatForSequenceClassification,
)
import numpy as np
import time
import pandas as pd
import torch
import librosa


device = "cuda"
durations = [1, 1, 5, 10, 20]
#durations = [10] #, 30]
#durations = [1]
og_rate = 8000
up_rate = 16000

upsample_algo = 'kaiser_best' # Other resampling techniques: https://librosa.org/doc/main/generated/librosa.resample.html

reps = 1000
    
pretrained_path = 'facebook/wav2vec2-large'
#pretrained_path = 'facebook/wav2vec2-xls-r-1b'
    
model = Wav2Vec2ForSequenceClassification.from_pretrained(
    pretrained_path,
    num_labels=4, 
)
model.to(device)
feature_extractor = Wav2Vec2FeatureExtractor()

for duration in tqdm(durations, desc='Duration'): 
    inference_results = []
    for i in tqdm(range(reps), desc='Reps'):
        # Generate a random wave of the desired duration
        raw_signal = np.random.rand(duration * og_rate)

        # Upsample to wanted frequency
        start_time = time.time()
        raw_signal = librosa.resample(raw_signal, og_rate, up_rate, res_type=upsample_algo)
        upsample_time = time.time() - start_time
 
        # Pre process file to input_values
        start_time = time.time()
        input_values = feature_extractor(raw_signal, sampling_rate=up_rate, return_tensors="pt").input_values
        pre_proc_time = time.time() - start_time

        # Move data to GPU
        start_time = time.time()
        input_values = input_values.to(device)
        data_2_gpu_time = time.time() - start_time

        # Infer label
        start_time = time.time()
        logits = model(input_values).logits
        cls_pred_ids = torch.argmax(logits, dim=-1)
        label = cls_pred_ids[0].item()
        classify_time = time.time() - start_time

        inference_results.append({
            'upsample_time': upsample_time,
            'pre_proccess' : pre_proc_time,
            'to_gpu'       : data_2_gpu_time,
            'classify'     : classify_time,
            'inference'    : pre_proc_time + data_2_gpu_time + classify_time,
            'y'            : label,
        })

    inf_res_df = pd.DataFrame(inference_results)
    #print(inf_res_df)
    print(str(duration) + ':')
    for column in inf_res_df:
        print(column + ': ' + str(np.mean(inf_res_df[column])))