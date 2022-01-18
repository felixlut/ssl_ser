### Based on the huggingface wav2vec2-ASR script (https://github.com/huggingface/transformers/blob/master/examples/research_projects/wav2vec2/run_asr.py)
import logging
import pathlib
import re
import sys
from types import SimpleNamespace
from pprint import pprint
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Union

import datasets
import numpy as np
import pandas as pd
import torch
from packaging import version
from torch import nn

import librosa
import wandb
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
    Wav2Vec2ForSequenceClassificationLinearHead,
    Wav2Vec2ForSequenceClassificationLinearHeadExtended,
    UniSpeechSatForSequenceClassificationLinearHead,
    UniSpeechSatForSequenceClassificationLinearHeadExtended,
    DataCollatorEmoWithPadding,
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
    from apex import amp 

if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_native_amp_available = True
    from torch.cuda.amp import autocast


logger = logging.getLogger(__name__)


def configure_logger(model_args: ModelArguments, training_args: TrainingArguments):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logging_level = logging.WARNING
    if model_args.verbose_logging:
        logging_level = logging.DEBUG
    elif trainer_utils.is_main_process(training_args.local_rank):
        logging_level = logging.INFO
    logger.setLevel(logging_level)


def main():
    ############# Settings ############# 
    model_type = Wav2Vec2ForSequenceClassification
    seed = 1337
    dataset_list = ['cremad', 'esd-eng', 'esd-chi', 'subesco', 'emodb', 'mesd', 'emovo', 'oreau', 'emouerj']
    label2id = {
        'Angry'  : 0, 
        'Happy'  : 1,
        'Sad'    : 2,
        'Neutral': 3,
    }
    label2id_merged, id2label = generate_reverse_dict(label2id)
    ####################################

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, DatasetArguments))
    model_args, data_args, training_args, dataset_args = parser.parse_args_into_dataclasses()

    configure_logger(model_args, training_args)

    data = Dataloader(data_args.data_path)
    datasets_to_load = ['cremad', 'esd', 'subesco', 'emodb', 'mesd', 'emovo', 'oreau', 'emouerj']
    train_df, test_df, val_df = partition_datasets(data, datasets_to_load, dataset_args, seed)
    train_df = train_df.loc[train_df.emo.isin(label2id.keys())]
    test_df = test_df.loc[test_df.emo.isin(label2id.keys())]
    val_df = val_df.loc[val_df.emo.isin(label2id.keys())]

    for subset in dataset_list:
        training_args.num_train_epochs = 20 if subset in ['cremad', 'esd-eng', 'esd-chi', 'subesco'] else 100
        model = model_type.from_pretrained(
            model_args.model_checkpoint,
            cache_dir=model_args.cache_dir,
            label2id=label2id,
            id2label=id2label,
            use_weighted_layer_sum=model_args.use_weighted_layer_sum, 
        )
        return_attention_mask = model.config.return_attention_mask if hasattr(model.config, 'return_attention_mask') else (model.config.feat_extract_norm == "layer")
        feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=False, return_attention_mask=return_attention_mask)

        # Freeze entire model, just a static embedder at that point
        if model_args.freeze_base_model:
            model.freeze_base_model()
        # Freeze only CNN-feature_extractor
        elif model_args.freeze_feature_extractor:
            model.freeze_feature_extractor()

        def prepare_sample(sample):
            signal, _ = librosa.load(sample[data_args.wav_column], sr=data_args.target_rate) 
            sample['speech'] = signal
            sample['labels'] = model.config.label2id[sample['emo']]
            return sample
        def prepare_dataset(batch):
            batch["input_values"] = feature_extractor(batch["speech"], sampling_rate=data_args.target_rate).input_values[0]
            return batch
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

        subset_train_df = train_df[train_df['dataset'] == subset]
        subset_test_df = test_df[test_df['dataset'] == subset]
        subset_val_df = val_df[val_df['dataset'] == subset]

        dataset_cache_dir = 'dataset_cache/'
        train_dataset = datasets.Dataset.from_pandas(subset_train_df)
        train_dataset.save_to_disk(dataset_cache_dir+'train/')
        train_dataset = datasets.load_from_disk(dataset_cache_dir+'train/')

        val_dataset = datasets.Dataset.from_pandas(subset_val_df)
        val_dataset.save_to_disk(dataset_cache_dir+'val/')
        val_dataset = datasets.load_from_disk(dataset_cache_dir+'val/')

        test_dataset = datasets.Dataset.from_pandas(subset_test_df)
        test_dataset.save_to_disk(dataset_cache_dir+'test/')
        test_dataset = datasets.load_from_disk(dataset_cache_dir+'test/')

        train_dataset = train_dataset.map(prepare_sample, remove_columns=[data_args.wav_column, 'emo'], desc='Training - Prep sample')
        train_dataset = train_dataset.map(prepare_dataset, num_proc=data_args.preprocessing_num_workers, remove_columns=['speech'], desc='Training - Prep dataset')
        val_dataset = val_dataset.map(prepare_sample, remove_columns=[data_args.wav_column, 'emo'], desc='Val - Prep sample')
        val_dataset = val_dataset.map(prepare_dataset, num_proc=data_args.preprocessing_num_workers, remove_columns=['speech'], desc='Val - Prep dataset')
        test_dataset = test_dataset.map(prepare_sample, remove_columns=[data_args.wav_column, 'emo'], desc='Test - Prep sample')
        test_dataset = test_dataset.map(prepare_dataset, num_proc=data_args.preprocessing_num_workers, remove_columns=['speech'], desc='Test - Prep dataset')

        data_collator = DataCollatorEmoWithPadding(feature_extractor=feature_extractor, padding=True)
        trainer = Trainer(
            model=model,
            data_collator=data_collator,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )
        trainer.train()

        wandb_log = 'wandb' in training_args.report_to

        # Log meta data of the datasets 
        [print_metadata(df, wandb_log=wandb_log, prefix=prefix) for df, prefix in zip([train_df, test_df, val_df], ['train_', 'test_', 'val_'])]

        # Evaluate overall performance
        output = trainer.predict(test_dataset, metric_key_prefix=subset)
        results = {
            'test/' + metric: output.metrics[metric] for metric in output.metrics
        } 
        pprint(results)

        axis = list(label2id_merged.keys())
        if wandb_log:
            wandb.log(results)
            report_result_confusion_matrix(output, axis, prefix='Overall_')
        
    if wandb_log:
        wandb.finish()

if __name__ == "__main__":
    main()