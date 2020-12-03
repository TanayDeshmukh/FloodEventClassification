import os
import re
import torch
import jsonlines
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tifffile import imread

class SentinalDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.sequences = []
        self.labels = []
        self.lables_json = root_dir+'devset_labels.jsonl'
        
        with jsonlines.open(self.lables_json) as reader:
            for obj in reader:
                path = self.root_dir+obj["sequence_id"]
                if os.path.exists(path):
                    self.sequences.append(obj["sequence_id"])
                    self.labels.append(obj["label"])
                    
    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):

        folder = str(self.sequences[idx])
        label = torch.tensor([self.labels[idx]]).type(torch.FloatTensor)

        blue = torch.from_numpy(imread(self.root_dir+folder+'/B02_series.tif').transpose().astype(np.float))
        green = torch.from_numpy(imread(self.root_dir+folder+'/B03_series.tif').transpose().astype(np.float))
        red = torch.from_numpy(imread(self.root_dir+folder+'/B04_series.tif').transpose().astype(np.float))

        if len(blue) == 512:
            blue = blue.unsqueeze(0)
            green = green.unsqueeze(0)
            red = red.unsqueeze(0)

        length = len(blue)

        sequence = torch.cat((red.unsqueeze(3), green.unsqueeze(3), blue.unsqueeze(3)), 3).type(torch.FloatTensor)

        sequence = torch.cat((sequence, torch.zeros([24 - sequence.size(0), 512, 512, 3]).type(torch.FloatTensor)))

        return sequence, label, length
