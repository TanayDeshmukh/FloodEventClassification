import torch

def train(model, data_loader, optimizer, criterion, device, epoch, num_epochs, summary, loss_display_interval):
    
    model.train()
    num_batches = len(data_loader)

    for i, (sequence, label, lengths) in enumerate(data_loader):
        
        optimizer.zero_grad()

        prediction, sort_indices = model(sequence, lengths)

        label = label[sort_indices]

        loss = criterion(prediction, label)

        if i % loss_display_interval == 0:
            step = epoch * num_batches + i
            print('TRAIN Epoch: [{}/{}], Batch Num: [{}/{}], Loss: [{}]'.format(epoch,num_epochs, i, num_batches, loss.item()))
            summary.add_scalar('Training loss', loss.item(), step)

        loss.backward()
        optimizer.step()
    
def validate(model, data_loader, optimizer, criterion, device, epoch, num_epochs, summary, loss_display_interval):
    
    model.eval()
    num_batches = len(data_loader)
    total_loss = 0

    for i, (sequence, label, lengths) in enumerate(data_loader):
        
        prediction, sort_indices = model(sequence, lengths)

        label = label[sort_indices]

        loss = criterion(prediction, label)

        if i % loss_display_interval == 0:
            step = epoch * num_batches + i
            print('VALIDATION Epoch: [{}/{}], Batch Num: [{}/{}], Loss: [{}]'.format(epoch,num_epochs, i, num_batches, loss.item()))
            summary.add_scalar('Validation loss', loss.item(), step)

        total_loss += loss.item()
    
    return total_loss/num_batches