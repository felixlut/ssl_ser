export MODEL=facebook/wav2vec2-large 

export RUN_NAME=mix_wav2-large_cremad_phone-vs-native
export UPLOAD_MODEL=false

export WAV_PATH=Nan # wav_path

export EPOCHS=20

export MULTI_LAYER=false # Careful using this one, the regular ModelForSequenceClassification will run into OOM-errors. See Wav2Vec2ForSequenceClassificationMultiLayer for quickfix
export FREEZE_FEATURE_EXTRACTOR=true
export FREEZE_BASE_MODEL=false

export PRESET_SPLIT=true # 70/15/15 split by default

export HOLDOUT_DATASET='nan' # esd-chi
export HOLDOUT_DATASET_SAMPLES=0
# 1(0s), 100(5m 32s), 500(27m 20s), 1000(54m 27s), 2000(1h 48m 13s), 4000(3h 34m 34s), 6000(5h 22m 0s), 8400 (7h 30m 34s)

export RUN_PATH=$PWD/ser_scripts
export DATASET_PATH=$RUN_PATH/datasets/

python $RUN_PATH/run_phone_test.py \
--model_checkpoint=$MODEL \
--output_dir=output/tmp \
--cache_dir=hf_cache/ \
--num_train_epochs=$EPOCHS \
--per_device_train_batch_size="4" \
--per_device_eval_batch_size="4" \
--eval_accumulation_steps="5" \
--evaluation_strategy="epoch" \
--save_strategy="epoch" \
--save_steps="500" \
--eval_steps="500" \
--save_total_limit="1" \
--logging_steps="50" \
--logging_dir="log" \
--learning_rate="1e-5" \
--lr_scheduler_type="linear" \
--fp16 true \
--preprocessing_num_workers="4" \
--gradient_checkpointing true \
--freeze_feature_extractor $FREEZE_FEATURE_EXTRACTOR \
--freeze_base_model $FREEZE_BASE_MODEL \
--dataloader_num_workers 2 \
--wav_column=$WAV_PATH \
--overwrite_output_dir true \
--report_to="wandb" \
--run_name=$RUN_NAME \
--metric_for_best_model="eval_WA" \
--load_best_model_at_end false \
--greater_is_better true \
--use_weighted_layer_sum $MULTI_LAYER \
--data_path $DATASET_PATH \
--holdout_dataset "$HOLDOUT_DATASET" \
--holdout_dataset_num_train $HOLDOUT_DATASET_SAMPLES \
--use_preset_split $PRESET_SPLIT \
--upload_model $UPLOAD_MODEL \
