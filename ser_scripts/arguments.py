from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_checkpoint: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    cache_dir: Optional[str] = field(
        #default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    freeze_feature_extractor: Optional[bool] = field(
        default=True, metadata={"help": "Whether to freeze the feature extractor layers of the model."}
    )
    freeze_base_model: Optional[bool] = field(
        default=False, metadata={"help": "Whether to freeze the entire model or not. Overrides freeze_feature_extractor"}
    )
    verbose_logging: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to log verbose messages or not."},
    )
    use_weighted_layer_sum: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use the outputs of all of the transformer-layers or just the last."},
    )
    upload_model: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to upload the model to wandb or not"},
    )
    
    

@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    data_path: str = field(
        metadata={"help": "The path to the datasets."}
    )
    holdout_dataset: Optional[str] = field(
        default="",
        metadata={"help": "Which dataset will be used as a holdout-set (wav_path or wav_tele_path). Defaults to not using a holdout-set"},
    )
    holdout_dataset_num_train: Optional[int] = field(
        default=0,
        metadata={"help": "How many samples from the holdout set to use during training. Defaults to 0, i.e. only for evaluation purposes."},
    )
    wav_column: Optional[str] = field(
        default="wav_tele_path",
        metadata={"help": "Column in the dataset that contains speech file path (wav_path or wav_tele_path). Defaults to 'wav_tele_path'"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    target_rate: Optional[int] = field(
        default=16000,
        metadata={"help": "The Sampling-rate of the sound-clips."},
    )


@dataclass
class DatasetArguments:
    k_fold: Optional[int] = field(
        default=5,
        metadata={"help": "The number of fold for k-fold cross-validation"},
    )
    train_split: Optional[float] = field(
        default=0.7,
        metadata={"help": "The size of the train partition in %"},
    )
    test_split: Optional[float] = field(
        default=0.15,
        metadata={"help": "The size of the test partition in %"},
    )
    val_split: Optional[float] = field(
        default=0.15,
        metadata={"help": "The size of the validation partition in %"},
    )
    use_preset_split: Optional[bool] = field(
        default=True, metadata={"help": "Whether to use the preset datasplits. This will override the split-percentages"}
    )
    iemocap_test_session: Optional[str] = field(
        default="1",
        metadata={"help": "Which of the sessions that will be used for testing"},
    )
    iemocap_val_session: Optional[str] = field(
        default="2",
        metadata={"help": "Which of the sessions that will be used for validation"},
    )
    esd_lang: Optional[str] = field(
        default="both",
        metadata={"help": "ESD contains samples of both eng and chi, pich which of these, or both to keep"},
    )