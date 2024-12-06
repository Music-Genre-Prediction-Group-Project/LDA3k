import torch
from torch.utils.data import Dataset
import pandas as pd

class LyricsDataset(Dataset):
    def __init__(self, csv_file):
        self.data_frame = pd.read_csv(csv_file)
        self.genres = self.data_frame['genre'].unique()
        self.genre_to_idx = {genre: idx for idx, genre in enumerate(self.genres)}
        self.idx_to_genre = {idx: genre for genre, idx in self.genre_to_idx.items()}

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        topics = self.data_frame.iloc[idx, 2:].values.astype('float32')
        genre = self.data_frame.iloc[idx, 1]
        label = self.genre_to_idx[genre]
        sample = {'topics': torch.tensor(topics), 'label': torch.tensor(label)}
        return sample