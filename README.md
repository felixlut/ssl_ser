# Speech Emotion Recognition based on pre-trained models using Self-Supervised Learning
This project is part of a master's thesis for the: course DA231X - Degree Project in Computer Science and Engineering, Second Cycle 30hp

Heavily influenced by Huggingface's ASR scripts for wav2vec2 (link [here](https://github.com/huggingface/transformers/tree/9a06b6b11bdfc42eea08fa91d0c737d1863c99e3/examples/research_projects/wav2vec2)). 

# Get started
Enter the following commands to the command line to setup and run the scripts
1. ```docker build -t ser_env .```
2. ```docker run -it --gpus device=0 -v $(pwd)/ser_scripts:/workspace/ser_scripts:ro ser_env```
    * To run a custom script instead of the regular run_training.sh as defined in the Dockerfile:
        * ```docker run -it --gpus device=0 -v $(pwd)/ser_scripts:/workspace/ser_scripts:ro ser_env bash ser_scripts/run_xval.sh ```

Change any parameters in ```run_x.sh``` to customize the run. The ```ser_scripts``` directory is loaded as a volume in the container, so any changes done locally will be seen in the container as well. 

# Files
There are 2 kinds of files, build-files and actual scripts. Build-files can be used to construct the environment for running the scripts.

## Build-files
* ***Dockerfile*** - Creates a basic container for running the different SER-experiments. The container runs ```run_training.sh``` by default
* ***requirements.txt*** - Lists the required packages for the environment.
    * One of the dependancies is the sister-repo of this project, which is made for constructing dataframes for the different datasets. It can be found [here](https://github.com/felixlut/SER_dataloader)

 
## Scripts

There are helper and run/test-scripts. Helper-scripts contains utility functions and structures which the run/test-scripts use as building-blocks to define the different experiments. 

### Helpers
* ***arguments.py*** - Contains dataclasses for parsing arguments to the main run-files. This allows for ease of use for running experiments from the command line
    * *ModelArguments* - Where to load/save model, freeze sections of model, ...
    * *DataArguments* - Where to load data from, sound-quality, ...
    * *DatasetArguments* - Train/test/validation splits, k-fold, ...
* ***data_partitioner.py*** - Partition the data into train/test/validation or k-fold splits
* ***model.py*** - Structs for different SSL-models and classification-heads
* ***utils.py*** - Utility scripts for plotting, graphs, ...

### Run/test-Scripts
Run-scripts have 2 parts, one .sh file meant to serve as a hub for setting the hyper-parameters of the run, while the python-file constructs the logic of the experiments. To run these enter ```bash run_x.sh``` (or ```bash ser_scripts/run_x.sh``` if in a container) to the command-line. Test-scripts are scripts not meant for any specific experiment, but more so for trying stuff out.  

Run:
* ***run_training*** - Basic script which the rest are built from
* ***run_baselines*** - Create mono-lingual baselines for each dataset
* ***run_phone_test*** - Test the difference of applying the phone-filter
* ***run_xval*** - Run cross-validation

Test:
* ***test_trained_model*** - Run an old model created by one of the run-scripts on custom data
* ***test_trained_model_wandb*** - Same as above, but load model from wandb (delete after I've merged the two)
* ***test_inference_time*** - Test how long different parts of the inference process take