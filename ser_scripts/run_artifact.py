### Based on the huggingface wav2vec2-ASR script (https://github.com/huggingface/transformers/blob/master/examples/research_projects/wav2vec2/run_asr.py)
import logging
import pathlib
import re
import sys
from types import SimpleNamespace
from pprint import pprint
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Union
import pandas as pd
import wandb

import datasets
import numpy as np
import torch
from packaging import version
from torch import nn

import librosa
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    AutoFeatureExtractor,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
    UniSpeechForSequenceClassification,
    UniSpeechSatForSequenceClassification,
    HubertForSequenceClassification,
    is_apex_available,
    trainer_utils,
)

from dataloader import Dataloader

from model import (
    Wav2Vec2ForSequenceClassificationMLP,
    UniSpeechForSequenceClassificationMLP,
    UniSpeechSatForSequenceClassificationMLP,
    DataCollatorEmoWithPadding,
    Wav2Vec2ForSequenceClassificationLinearHead,
    Wav2Vec2ForSequenceClassificationLinearHeadExtended,
)
from arguments import ModelArguments, DataArguments, DatasetArguments
from utils import (
    generate_reverse_dict,
    compute_WA_UA,
    print_metadata,
    report_result_confusion_matrix,
)
from data_partitioner import partition_datasets


if is_apex_available():
    from apex import amp # TODO: Check again

if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_native_amp_available = True
    from torch.cuda.amp import autocast

###########################################################

label2id = {
    'Angry'  : 0, 
    'Happy'  : 1,
    #'Excited': 1,
    'Sad'    : 2,
    'Neutral': 3,
}
label2id_merged, id2label = generate_reverse_dict(label2id)

run = wandb.init()
run_path = 'fellut/huggingface/7q0pqdjg'
artifact = run.use_artifact(run_path+':v0', type='model')
artifact_dir = artifact.download()

data_splits = ['val', 'test']
model_checkpoints = ['119600']
seed = 1337
cache_dir='hf_cache/'
base_model = 'facebook/wav2vec2-large-xlsr-53'

top_data_path = '/workspace/ser_scripts/datasets/'
data = Dataloader(top_data_path)

dataset_list = ['cremad', 'emodb', 'emouerj', 'emovo', 'esd', 'mesd', 'oreau', 'subesco']
#dataset_list = ['esd']
holdout_dataset_name = 'esd-chi' # Remove this part to run all

dataset_args = DatasetArguments(use_preset_split=True)

train_df, test_df, val_df = partition_datasets(data, dataset_list, dataset_args)
train_df = train_df.loc[train_df.emo.isin(label2id.keys())]
test_df = test_df.loc[test_df.emo.isin(label2id.keys())]
val_df = val_df.loc[val_df.emo.isin(label2id.keys())]

train_df = train_df.loc[train_df['dataset'] == holdout_dataset_name]
test_df = test_df.loc[test_df['dataset'] == holdout_dataset_name]
val_df = val_df.loc[val_df['dataset'] == holdout_dataset_name]

dataset_df_by_split = {
    'train': train_df,
    'test': test_df,
    'val': val_df,
}

def compute_metrics(pred):
    logits = pred.predictions
    pred_ids = np.argmax(logits, axis=-1)
    total = len(pred.label_ids)
    correct = (pred_ids == pred.label_ids).sum()

    WA, UA, acc_by_label = compute_WA_UA(pred_ids, pred.label_ids, label2id_merged)
    acc_by_label['WA'] = WA
    acc_by_label['UA'] = UA
    acc_by_label['correct'] = correct
    acc_by_label['total'] = total
    return acc_by_label

def prepare_sample(sample):
    signal, rate = librosa.load(sample['wav_tele_path'], sr=16000) # TODO: Fix hardcode
    sample['speech'] = signal
    sample['labels'] = label2id[sample['emo']]
    return sample

def prepare_dataset(batch):
    batch["input_values"] = np.squeeze(feature_extractor(batch["speech"], sampling_rate=16000).input_values)
    return batch

model_for_settings = Wav2Vec2ForSequenceClassificationLinearHeadExtended.from_pretrained(base_model)
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=False, return_attention_mask=(model_for_settings.config.feat_extract_norm == "layer"))
#feature_extractor = AutoFeatureExtractor.from_pretrained(artifact_dir, do_normalize=False) 

for split in data_splits:
    df = dataset_df_by_split[split]
    dataset = datasets.Dataset.from_pandas(df)
    dataset.save_to_disk('dataset_delete_when_done/')
    dataset = datasets.load_from_disk('dataset_delete_when_done/')

    dataset = dataset.map(prepare_sample, remove_columns=['wav_tele_path', 'emo'], desc='Prep sample')
    dataset = dataset.map(prepare_dataset, num_proc=4, remove_columns=['speech'], desc='Prep dataset')
    
    for checkpoint_id in model_checkpoints:
        model_dir = artifact_dir + '/checkpoint-'+checkpoint_id
        model = Wav2Vec2ForSequenceClassification.from_pretrained(
            model_dir,
            label2id=label2id,
            id2label=id2label,
        )
        data_collator = DataCollatorEmoWithPadding(feature_extractor=feature_extractor, padding=True)

        training_args = TrainingArguments('output/tmp2', per_device_eval_batch_size=1)
        trainer = Trainer(
            model=model,
            data_collator=data_collator,
            args=training_args,
            compute_metrics=compute_metrics,
        )

        axis = list(label2id_merged.keys())
        subsets = set(dataset['dataset'])
        output = trainer.predict(dataset, metric_key_prefix=split)
        results = {
            checkpoint_id+'/'+split+'/'+metric: + output.metrics[metric] for metric in output.metrics
        } 
        pprint(results)
        for subset in subsets:
            subset_indexes = [i for i in range(len(dataset)) if dataset[i]['dataset'] == subset]
            subset_pred = output.predictions[subset_indexes]
            subset_labels = output.label_ids[subset_indexes]
            
            args = {
                'predictions': subset_pred,
                'label_ids': subset_labels,
            }
            compute_input = SimpleNamespace(**args)
            compute_output = compute_metrics(compute_input)
            results = {
                checkpoint_id+'/'+split+'/'+subset+'_'+metric: + compute_output[metric] for metric in compute_output
            }
            pprint(results)
