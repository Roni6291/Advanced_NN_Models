from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np


class TimeSeriesDataset(Dataset):

    def __init__(self, df, seq_len):
        self.data = df.values
        self.seq_len = seq_len
        self.features = None
        self.labels = None

    def __len__(self):
        return self.data.shape[0] // self.seq_len

    def __getitem__(self, idx, features=None, labels=None):

        if features is None:
            self.features = []

        if labels is None:
            self.labels = []

        for rec in range(len(self.data)):
            end_ix = rec + self.seq_len

            if end_ix > len(self.data) - 1:
                break

            self.features.append(self.data[rec:end_ix])
            self.labels.append(self.data[end_ix])

        return self.features[idx], self.labels[idx]


df = pd.DataFrame(data=np.arange(0, 60).reshape(
    20, 3), columns=['A', 'B', 'C'])
print(df)

dataset = TimeSeriesDataset(df, 4)

loader = DataLoader(dataset, batch_size=1, shuffle=False)

for batch, (feature, label) in enumerate(loader):
    print(feature)
    print(label)
    # print(feature.shape)
    # print(label.shape)
    # break
