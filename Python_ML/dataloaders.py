from torch.utils.data import Dataset


class SoundDS(Dataset):
    # Add Classes to dataset
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        class_id = self.df.loc[idx, 'Class']
        spec = self.df.loc[idx, 'Spectrogram']
        return spec, class_id
