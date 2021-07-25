from data_preprocessing import preprocess_all_data
from torch.utils.data import random_split
from dataloaders import SoundDS
import torch


def create_dataset(test_size):
    # Load all dataset songs
    songs = ['ADIZ', 'JLEE', 'JTAN', 'KENN', 'MCUR', 'MPOL', 'MPUR', 'NJAT', 'PMAR', 'SAMF', 'VKOW', 'ZHIY']
    # Hyper - parameters
    audio_channels = 1  # 1 for Mono Recordings
    new_sr = 44100  # Sampling rate to be used for all songs
    duration_ms = 2000  # Signal Duration in ms
    min_avg = 0.0015  # Minimum Average to be Considered Silence (experimentally obtained value based on the dataset)

    # Create dataframe
    data_df = preprocess_all_data(songs, audio_channels, new_sr, duration_ms, min_avg)
    # Create dataset with classes
    myds = SoundDS(data_df)
    # Random split of 80:20 between training and validation
    num_items = len(myds)
    num_train = round(num_items * (1.-test_size))
    num_val = num_items - num_train
    train_ds, val_ds = random_split(myds, [num_train, num_val])

    # Create training and validation data loaders
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=16, shuffle=False)
    return train_dl, val_dl
