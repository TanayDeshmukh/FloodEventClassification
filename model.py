import torch.nn as nn
import torchvision
import torch
import numpy as np
from tifffile import TiffFile, TiffSequence, imread
import matplotlib.pyplot as plt

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

class myLSTM(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, batch_size):
        super(myLSTM, self).__init__()

        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(512, 128)
        self.batch_size = batch_size
        self.dropout = nn.Dropout(p=0.2)
        self.decode_step = nn.LSTMCell(encoder_dim, decoder_dim, bias=True)

        self.linear1 = nn.Linear(128, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 1)
        
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fine_tune()      

    def fine_tune(self, fine_tune=True):

        for p in self.resnet.parameters():
            p.requires_grad = False

        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune

    def forward(self, sequence, lengths):

        h = torch.zeros([self.batch_size, 128]).to(device)
        c = torch.zeros([self.batch_size, 128]).to(device)

        lengths, sort_indices = lengths.sort(descending=True)
        
        sequence = sequence[sort_indices]
        sequence = torch.transpose(sequence, 2, 4).to(device)

        hidden = torch.zeros([self.batch_size, 128])

        for i in range(max(lengths)):
            batch_size_i = sum([l > i for l in lengths]).item()
            resnet_out = self.resnet(sequence[:batch_size_i, i, :])
            h, c = self.decode_step(resnet_out, (h[:batch_size_i], c[:batch_size_i]))

            hidden[:batch_size_i] = h[:batch_size_i]

        out = self.relu(self.linear1(hidden))
        out = self.dropout(out)
        out = self.relu(self.linear2(out))
        out = self.sigmoid(self.linear3(out))

        return out, sort_indices
