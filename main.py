import torch
import torch.optim

import numpy as np
import torch.nn as nn

from model import myLSTM
from train import train, validate
from dataLoader import SentinalDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import generate_train_validation_samplers, adjust_learning_rate, save_checkpoint

root_folder = './data/'
start_epoch = 0
epochs_since_improvement = 0
epochs = 100
batch_size = 4
learning_rate = 4e-4
grad_clip = 4.

loss_display_interval = 2
hidden_dimension = 128
resnet_out_dimension = 128
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

checkpoint = None
min_loss = np.inf

def main():

    global root_folder, start_epoch, epochs_since_improvement, epochs, batch_size, learning_rate, grad_clip, loss_display_interval, min_loss

    if checkpoint is None:
        model = myLSTM(resnet_out_dimension, hidden_dimension, batch_size).float()
        optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)


    model.to(device)

    data_set = SentinalDataset(root_folder+'train/')
    train_sampler, validation_sampler = generate_train_validation_samplers(data_set, validation_split=0.2)

    train_data_loader = DataLoader(data_set, batch_size = batch_size, sampler = train_sampler, drop_last = True)
    validation_data_loader = DataLoader(data_set, batch_size = batch_size, sampler = validation_sampler, drop_last = True)

    criterion = nn.BCELoss()
    summary = SummaryWriter()

    for epoch in range(start_epoch, epochs):

        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(optimizer, 0.8)

        train(model, train_data_loader, optimizer, criterion, device, epoch, epochs, summary, loss_display_interval)
        validation_loss = validate(model, validation_data_loader, optimizer, criterion, device, epoch, epochs, summary, loss_display_interval)
        is_best = validation_loss<min_loss
        min_loss = min(min_loss, validation_loss)
        
        if not is_best:
            epochs_since_improvement += 1
        else:
            epochs_since_improvement = 0
        
        save_checkpoint(epoch, epochs_since_improvement, model, optimizer, validation_loss, is_best)

    summary.flush()

if __name__ == '__main__':
    main()