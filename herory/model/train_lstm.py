import torch
from torch import nn, optim
import random
import time
import math
import os

def train_lstm_with_masking(
    model, 
    dataloader, 
    criterion=None, 
    optimizer=None, 
    sequence_length=128, 
    epochs=10,
    device='cpu',
    masking_prob=0.05,
    clipping_value=1,
    last_epoch=0,
    checkpoints_dir='./saves/lstm/',
    epoch_save_point=10,
    ):
    '''
        Train LSTM model with model, dataloader, criterion [optional], optimizer [optional], 
        sequence length and number of epochs
        
        Training with masking `<unk>` token

        Inputs:
            - model: model to train
            - dataloader: torch data loader
            - criterion: torch criterion (default: cross entropy loss)
            - optimizer: torch optimizer (default: adam with learning rate = 0.001)
            - sequence_length: sequence length (default: 128)
            - epochs: number of epochs
            - device (string, default: `cpu`): device {`cpu` or `cuda`}
            - masking_prob (float, default: 0.05): probability to mask a word in training
            - clipping_value (float, default: `1`): gradient clipping value, ranged at [-clipping_value, clipping_value]
            - last_epoch (integer, default: `-1`): last epoch the model trained on, not trained yet: -1
            - checkpoints_dir (string, default: `./saves/lstm/`): directory of the checkpoints to save
            - epoch_save_point (integer, default: `10`): save point of checkpoint
    '''
    start_time = time.time()
    _batches = len(dataloader)
    
    if criterion == None:
        criterion = nn.CrossEntropyLoss().to(device)

    if optimizer == None:
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
    model.to(device)
    model.train()
    
    losses = []
    avg_losses = []
    
    _this_epoch = last_epoch + 1
    
    for epoch in range(_this_epoch, _this_epoch + epochs):
        state_h, state_c = model.init_state(sequence_length)
        
        _epoch_losses = []

        for batch, (x, y) in enumerate(dataloader):
            
            # generate mask
            _mask_seeds = torch.rand(x.shape)
            mask = _mask_seeds.le(masking_prob)
            
            # masking inputs
            masked_x = torch.masked_fill(x, mask, 0)
            
            masked_x = masked_x.to(device)
            y = y.to(device)

            y_pred, (state_h, state_c) = model(masked_x, (state_h, state_c))
            loss = criterion(y_pred.transpose(1, 2), y)

            state_h = state_h.detach()
            state_c = state_c.detach()

            optimizer.zero_grad()
            loss.backward()
            
            # gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), clipping_value)
            optimizer.step()
            
            losses.append(loss.item())
            _epoch_losses.append(loss.item())
            
            if batch % 1000 == 0:
                print({ 'epoch': epoch, 'batch': batch, 'loss': loss.item(), 'epoch_progress[%]': math.floor(batch / _batches * 100) })
        
        _loss_sum = 0
        for _epoch_loss in _epoch_losses:
            _loss_sum += _epoch_loss
        _avg_loss = _loss_sum / _batches
        avg_losses.append(_avg_loss)
        
        if (epoch + 1) % epoch_save_point == 0:
            print('saving checkpoints ...')
            torch.save(model.state_dict(), os.path.join(checkpoints_dir, f'{epoch}-checkpoint.pth'))
            torch.save(losses[-epoch_save_point*_batches:], os.path.join(checkpoints_dir, f'{epoch}-losses.pt'))
            torch.save(avg_losses[-epoch_save_point:], os.path.join(checkpoints_dir, f'{epoch}-avg_losses.pt'))
            torch.save(model.state_dict(), os.path.join(checkpoints_dir, 'lastest.pth'))
            print(f'checkpoints saved to {checkpoints_dir}')
    
    print(f'Training time: {time.time() - start_time} s')

    return {'losses': losses, 'avg_losses': avg_losses, 'last_epoch': epochs+last_epoch}

def train_seq2seq_with_masking(
    model, 
    dataloader, 
    tokenizer, 
    criterion=None, 
    optimizer=None, 
    epochs=10,
    device='cpu',
    masking_prob=0.05,
    clipping_value=1,
    last_epoch=0,
    checkpoints_dir='./saves/seq2seq/',
    epoch_save_point=10,
    ):
    '''
        Train sequence-to-sequence / encoder-decoder model with model, dataloader, criterion [optional], optimizer [optional], 
        sequence length and number of epochs
        
        Training with masking `<unk>` token

        Inputs:
            - model: model to train
            - dataloader: torch data loader
            - tokenizer: HERORY tokenizer
            - criterion: torch criterion (default: cross entropy loss)
            - optimizer: torch optimizer (default: SGD with learning rate = 0.001)
            - epochs: number of epochs
            - device (string, default: `cpu`): device {`cpu` or `cuda`}
            - masking_prob (float, default: 0.05): probability to mask a word in training
            - clipping_value (float, default: `1`): gradient clipping value, ranged at [-clipping_value, clipping_value]
            - last_epoch (integer, default: `0`): last epoch the model trained on
            - checkpoints_dir (string, default: `./saves/seq2seq/`): directory of the checkpoints to save
            - epoch_save_point (integer, default: `10`): save point of checkpoint
    '''
    start_time = time.time()
    _batches = len(dataloader)
    
    if criterion == None:
        criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.get_padding_token_index())

    if optimizer == None:
        optimizer = optim.SGD(model.parameters(), lr=0.001)
        
    model.to(device)
    model.train()
    
    losses = []
    avg_losses = []
    
    for epoch in range(last_epoch, last_epoch + epochs):
        
        _epoch_losses = []
        
        for batch, (x, y) in enumerate(dataloader):
            
            # generate mask
            _mask_seeds = torch.rand(x.shape)
            mask = _mask_seeds.le(masking_prob)
            
            # masking inputs
            masked_x = torch.masked_fill(x, mask, 0)
            
            masked_x = masked_x.to(device)
            
            _padding = torch.tensor([tokenizer.get_padding_token_index()])
            padding = _padding.repeat(y.shape[0], 1)
            # padding: [batch_size, 1]
            
            y = torch.cat([padding, y], dim=1)
            # y: [batch_size, sequence_length + 1]
            y = y.to(device)
            
            output, _ = model(masked_x, y)
            # output: [sequence_length + 1, batch_size, num_vocab]
            
            output = output.permute(1, 2, 0)
            # output: [batch_size, num_vocab, sequence_length + 1]
            
            loss = criterion(output, y)
            
            optimizer.zero_grad()
            loss.backward()
            
            # gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), clipping_value)
            optimizer.step()
            
            losses.append(loss.item())
            _epoch_losses.append(loss.item())
        
            if batch % 1000 == 0:
                print({ 'epoch': epoch, 'batch': batch, 'loss': loss.item(), 'epoch_progress[%]': math.floor(batch / _batches * 100) })
        
        
        _loss_sum = 0
        for _epoch_loss in _epoch_losses:
            _loss_sum += _epoch_loss
        _avg_loss = _loss_sum / _batches
        avg_losses.append(_avg_loss)
        
        if (epoch + 1) % epoch_save_point == 0:
            print('saving checkpoints ...')
            torch.save(model.state_dict(), os.path.join(checkpoints_dir, f'{epoch}-checkpoint.pth'))
            torch.save(losses[epoch+1-epoch_save_point:], os.path.join(checkpoints_dir, f'{epoch}-losses.pt'))
            torch.save(avg_losses[epoch+1-epoch_save_point:], os.path.join(checkpoints_dir, f'{epoch}-avg_losses.pt'))
            torch.save(model.state_dict(), os.path.join(checkpoints_dir, 'lastest.pth'))
            print(f'checkpoints saved to {checkpoints_dir}')
    
    print(f'Training time: {time.time() - start_time} s')

    return {'losses': losses, 'avg_losses': avg_losses, 'last_epoch': epochs-1+last_epoch}
