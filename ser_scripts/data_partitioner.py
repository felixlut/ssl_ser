import pandas as pd
import numpy as np
import itertools
import math

def _load_preset_dataset_partitions(data, subset, dataset_args):
    # Retrieve the preset split for the wanted dataset. Splits stored in datasets/dataset_name/split.csv
    data.load_dataset(subset)
    path = getattr(data, subset).path
    if subset != 'esd':
        train_df, test_df, val_df = (pd.read_csv(path + split + '.csv', squeeze=True) for split in ['train', 'test', 'val'])
    else:
        langs = ['eng', 'chi'] if dataset_args.esd_lang == 'both' else [dataset_args.esd_lang]
        data_by_lang = []
        for lang in langs:
            train_df, test_df, val_df = (pd.read_csv(path + split + '_' + lang + '.csv', squeeze=True) for split in ['train', 'test', 'val'])
            data_by_lang.append({
                'train': train_df,
                'test': test_df,
                'val': val_df,
            })
        train_df, test_df, val_df = (pd.concat([data_dict[split] for data_dict in data_by_lang]) for split in ['train', 'test', 'val'])
        
    return train_df, test_df, val_df


def naive_split(df, train, test, val, seed):
    # Randomly assign samples to train-, test- and val-sets by the given percentages

    assert train+test+val == 1, 'Splits have to add up to 1'
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    cuts = list(itertools.accumulate([0, train, test, val]))
    cuts = [math.ceil(len(df)*cut) for cut in cuts]
    train_df, test_df, val_df = (df[cuts[i]:cuts[i+1]] for i in range(3))
    
    return train_df, test_df, val_df


def _random_actor_split(df, train, test, val, seed):
    # Randomly assign actors to train-, test- and val-sets by the given percentages
    assert train+test+val == 1, 'Splits has to add up to 1'
    assert train > test+val, 'Train-split > 50%'

    actors = list(set(df['actor_id']))

    np.random.seed(seed)
    np.random.shuffle(actors)

    # Catch the case that there are only 3 actors (EMOVO on gender)
    if len(actors) == 3:
        return (df.loc[df.actor_id == act] for act in actors)
    
    cuts = list(itertools.accumulate([0, val, test, train]))
    cuts = [math.ceil(len(actors)*cut) for cut in cuts]

    val_actors, test_actors, train_actors = (actors[cuts[i]:cuts[i+1]] for i in range(3))

    train_df, test_df, val_df = (df.loc[df.actor_id.isin(act)] for act in [train_actors, test_actors, val_actors])
    return train_df, test_df, val_df


def _split_by_col(df, col, split_function, train, test, val, seed):
    # Split the dataframe on col before running the split_function. Usually used for splitting on gender before actor_ids
    options = set(df[col])
    data_by_col = []
    for df_by_col in [df.loc[df[col] == value] for value in options]:
        train_df, test_df, val_df = split_function(df_by_col, train, test, val, seed)
        data_by_col.append({
            'train': train_df,
            'test': test_df,
            'val': val_df,
        })
    return (pd.concat([data_dict[split] for data_dict in data_by_col]) for split in ['train', 'test', 'val'])


def _load_dataset_partitions(data, subset, dataset_args, seed):
    # Split the dataset by gender, actor_ids and language, depending on which dataset it is

    subset_df = data.load_dataset(subset)
    if subset in ['cremad', 'oreau', 'ravdess', 'subesco', 'emodb', 'mesd', 'emovo', 'emouerj']:
        # Split on gender before partitioning samples to the different splits

        split_function = _random_actor_split if subset != 'mesd' else naive_split
        train_df, test_df, val_df = _split_by_col(subset_df, 'gender', split_function,
                                                    dataset_args.train_split, dataset_args.test_split, dataset_args.val_split, seed)

    elif subset == 'esd':
        # Specific split for ESD, since it's multilingual
        langs = ['eng', 'chi'] if dataset_args.esd_lang == 'both' else [dataset_args.esd_lang]
        data_by_lang = []
        for lang in langs:
            # Extract language dataframe
            lang_df = subset_df.loc[subset_df['lang'] == lang]
            split_function = _random_actor_split
            train_df, test_df, val_df = _split_by_col(lang_df, 'gender', split_function,
                                            dataset_args.train_split, dataset_args.test_split, dataset_args.val_split, seed)
            data_by_lang.append({
                'train': train_df,
                'test': test_df,
                'val': val_df,
            })
                
        train_df, test_df, val_df = (pd.concat([data_dict[split] for data_dict in data_by_lang]) for split in ['train', 'test', 'val'])

    else:
        raise NotImplementedError("{} not implemented".format(subset))

    return train_df, test_df, val_df


def partition_datasets(data, dataset_list, dataset_args, seed):
    # Load the data for the sets listed in dataset_list, either from the preset data-splits, or randomly split 

    partitioned_data = []
    for subset in dataset_list:
        if dataset_args.use_preset_split:
            train_df, test_df, val_df = _load_preset_dataset_partitions(data, subset, dataset_args)

            # TODO: Proper fix
            bad_path = 'home/fellut/speech_emotion_recognition/datasets/'
            good_path = data.dataset_path
            train_df['wav_tele_path'] = train_df['wav_tele_path'].apply(lambda x: x.replace(bad_path, good_path))
            test_df['wav_tele_path'] = test_df['wav_tele_path'].apply(lambda x: x.replace(bad_path, good_path))
            val_df['wav_tele_path'] = val_df['wav_tele_path'].apply(lambda x: x.replace(bad_path, good_path))

            train_df['wav_path'] = train_df['wav_path'].apply(lambda x: x.replace(bad_path, good_path))
            test_df['wav_path'] = test_df['wav_path'].apply(lambda x: x.replace(bad_path, good_path))
            val_df['wav_path'] = val_df['wav_path'].apply(lambda x: x.replace(bad_path, good_path))

        else:
            train_df, test_df, val_df = _load_dataset_partitions(data, subset, dataset_args, seed)
        
        
        partitioned_data.append({
            'train': train_df,
            'test': test_df,
            'val': val_df,
        })

    # Create supersets with only columns present in all datasets
    return (pd.concat([data_dict[split] for data_dict in partitioned_data], join="inner", axis=0, ignore_index=True) for split in ['train', 'test', 'val'])



def k_fold_naive_split(df, k, seed):
    # Randomly assign samples to each split
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    cuts = [i * 1/k for i in range(k + 1)]
    cuts = [math.ceil(len(df)*cut) for cut in cuts]
    
    splits = [df[cuts[i]:cuts[i+1]] for i in range(len(cuts) - 1)]
    
    return splits


def k_fold_actor_split(df, k, seed):
    # Randomly assign actors to each split
    actors = list(set(df['actor_id']))

    np.random.shuffle(actors)

    cuts = [i * 1/k for i in range(k + 1)]
    cuts = [math.ceil(len(actors)*cut) for cut in cuts]

    actors_by_split = [actors[cuts[i]:cuts[i+1]] for i in range(len(cuts) - 1)]
    splits = [df.loc[df.actor_id.isin(act)] for act in actors_by_split]

    return splits


def _load_dataset_partitions_xval(data, subset, dataset_args, seed):
    # k-fold split the dataset by gender, actor_ids and language, depending on which dataset it is

    subset_df = data.load_dataset(subset)

    if subset in ['cremad', 'oreau', 'ravdess', 'subesco', 'emodb', 'mesd']:
        split_by_gender = {}
        genders = set(subset_df['gender'])
        for gender in genders:
            df_by_gender = subset_df.loc[subset_df['gender'] == gender]

            # If the dataset have actor_ids, split on those. Otherwise naive split
            split_function = k_fold_actor_split if 'actor_id' in df_by_gender.columns else k_fold_naive_split
            split_by_gender[gender] = split_function(df_by_gender, dataset_args.k_fold, seed)

        splits = [pd.concat([split_by_gender[gender][fold] for gender in genders]) for fold in range(dataset_args.k_fold)]

    elif subset == 'esd':
        # Specific split for ESD, since it's multilingual
        langs = ['eng', 'chi'] if dataset_args.esd_lang == 'both' else [dataset_args.esd_lang]
        split_by_lang = {}
        for lang in langs:
            df_by_lang = subset_df.loc[subset_df['lang'] == lang]
            split_by_gender = {}
            genders = set(df_by_lang['gender'])
            for gender in genders:
                df_by_gender = df_by_lang.loc[df_by_lang['gender'] == gender]

                # If the dataset have actor_ids, split on those. Otherwise naive split
                split_function = k_fold_actor_split if 'actor_id' in df_by_gender.columns else k_fold_naive_split
                split_by_gender[gender] = split_function(df_by_gender, dataset_args.k_fold, seed)

            split_by_lang[lang] = [pd.concat([split_by_gender[gender][fold] for gender in genders]) for fold in range(dataset_args.k_fold)]

        splits = [pd.concat([split_by_lang[lang][fold] for lang in langs]) for fold in range(dataset_args.k_fold)]
    
    # EMOVO and emoUERJ skips the gender split, since they consist of so few actors
    elif subset in ['emovo', 'emouerj']:
        splits = k_fold_actor_split(subset_df, dataset_args.k_fold, seed)
        
    else:
        raise NotImplementedError("{} not implemented".format(subset))

    return splits


def partition_datasets_xval(data, dataset_list, dataset_args, seed):
    # Load the data for the sets listed in dataset_list, and split it into k folds

    np.random.seed(seed)

    splits_by_subset = {}
    for subset in dataset_list:
        splits_by_subset[subset] = _load_dataset_partitions_xval(data, subset, dataset_args, seed)
    
    splits = [pd.concat([splits_by_subset[subset][fold] for subset in dataset_list], join="inner", axis=0, ignore_index=True) 
                                                                                                for fold in range(dataset_args.k_fold)]
    return splits
