import torch
from torch import nn
import random

class LSTM(nn.Module):
    '''
        HERORY LSTM model
        
        Arguments:
            - num_vocab (integer): number of vocabulary
            - hidden_size (integer, default: `128`): hidden size of LSTM
            - embedding_dim (integer, default: `128`): embedding dimensions of embedding layer
            - num_layers (integer, default: `2`): number of layers for LSTM
            - dropout (float, default: `0.2`): dropout rate of LSTM
            - device (string, default: `cpu`): device {`cpu` or `cuda`}
    '''
    def __init__(
        self, 
        num_vocab, 
        hidden_size=256, 
        embedding_dim=128, 
        num_layers=2,
        dropout=0.2,
        device='cpu'
        ):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.device = device
        self.dropout = dropout

        self.embedding = nn.Embedding(
            num_embeddings=num_vocab,
            embedding_dim=self.embedding_dim,
        )
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )
        self.fc = nn.Linear(self.hidden_size, num_vocab)

    def forward(self, x, prev_state):
        embed = self.embedding(x)
        output, state = self.lstm(embed, prev_state)
        logits = self.fc(output)
        return logits, state

    def init_state(self, sequence_length):
        return (torch.zeros(self.num_layers, sequence_length, self.hidden_size).to(self.device),
                torch.zeros(self.num_layers, sequence_length, self.hidden_size).to(self.device))
        
class LSTMEncoder(nn.Module):
    '''
        HERORY LSTM encoder for seq2seq attention model
        
        Arguments:
            - num_vocab (integer): number of vocabulary
            - embedding_dim (integer): embedding dimensions of embedding layer
            - hidden_size (integer): hidden size of LSTM
            - dropout (float): dropout rate of LSTM
    '''
    def __init__(self, num_vocab, embedding_dim, hidden_size, dropout):
        super(LSTMEncoder, self).__init__()
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(
            num_embeddings=num_vocab,
            embedding_dim=embedding_dim,
        )
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=1,
            bidirectional=True
        )
        
        self.fc_h = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.fc_c = nn.Linear(self.hidden_size * 2, self.hidden_size)
        
    def forward(self, x):
        # x: [batch_size, sequence_length]
        x = x.transpose(0, 1)
        # x: [sequence_length, batch_size]
        
        embed = self.embedding(x)
        embed = self.dropout(embed)
        output, (h_t, c_t) = self.lstm(embed)
        
        h_t = self.fc_h(torch.cat((h_t[0: 1], h_t[1: 2]), dim=2))
        c_t = self.fc_c(torch.cat((c_t[0: 1], c_t[1: 2]), dim=2))
        
        return output, (h_t, c_t)

class LSTMDecoder(nn.Module):
    '''
        HERORY LSTM decoder for seq2seq attention model
        
        Arguments:
            - num_vocab (integer): number of vocabulary
            - embedding_dim (integer): embedding dimensions of embedding layer
            - hidden_size (integer): hidden size of LSTM
            - dropout (float): dropout rate of LSTM
    '''
    def __init__(self, num_vocab, embedding_dim, hidden_size, dropout):
        super(LSTMDecoder, self).__init__()
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(
            num_embeddings=num_vocab,
            embedding_dim=embedding_dim,
        )
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=self.hidden_size*2+embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=1
        )
        
        self.e = nn.Linear(self.hidden_size * 3, 1)
        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(self.hidden_size, num_vocab)
        
    def forward(self, x, encoder_output, h_t, c_t):
        # x: [batch_size]
        x = x.unsqueeze(0)
        # x: [1, batch_size]
        
        embed = self.embedding(x)
        # embed: [1, batch_size, embedding_dim]
        embed = self.dropout(embed)
        
        sequence_length = encoder_output.shape[0]
        hidden = h_t.repeat(sequence_length, 1, 1)
        
        energy = self.e(torch.cat((hidden, encoder_output), dim=2))
        energy = self.relu(energy)
        
        attention_weights = self.softmax(energy)
        
        context = torch.bmm(
            attention_weights.permute(1, 2, 0),
            encoder_output.permute(1, 0, 2)
        ).permute(1, 0, 2)
        # context: [1, batch_size, hidden_size * 2]
        
        lstm_input = torch.cat((context, embed), dim=2)
        
        output, (h_t, c_t) = self.lstm(lstm_input, (h_t, c_t))
        # output: [1, batch_size, hidden_size]
        
        output = self.fc(output)
        # output: [1, batch_size, num_vocab]
        
        output = output.squeeze(0)
        
        return output, (h_t, c_t), attention_weights

class AttnSeq2SeqLSTM(nn.Module):
    '''
        HERORY sequence-to-sequence / encoder-decoder model with attention mechanism
        
        LSTM encoder -> LSTM decoder with attention mechanism
        
        Arguments:
            - num_vocab (integer): number of vocabulary
            - embedding_dim (integer, default: `128`): embedding dimensions of embedding layer
            - hidden_size (integer, default: `128`): hidden size of LSTM encoder and decoder
            - dropout (float, default: `0.5`): dropout rate of LSTM encoder and decoder
            - device (string, default: `cpu`): device of model use on
            - teacher_force_ratio (float, default: `0.5`): teacher force ratio
    '''
    def __init__(
        self, 
        num_vocab, 
        embedding_dim=128, 
        hidden_size=128, 
        dropout=0.5, 
        device='cpu', 
        teacher_force_ratio=0.5
        ):
        super(AttnSeq2SeqLSTM, self).__init__()
        self.encoder = LSTMEncoder(
            num_vocab=num_vocab, 
            embedding_dim=embedding_dim, 
            hidden_size=hidden_size, 
            dropout=dropout
        ).to(device)
        self.decoder = LSTMDecoder(
            num_vocab=num_vocab, 
            embedding_dim=embedding_dim, 
            hidden_size=hidden_size, 
            dropout=dropout
        ).to(device)
        self.num_vocab = num_vocab
        self.device = device
        self.teacher_force_ratio = teacher_force_ratio
        
    def forward(self, x, y):
        # x: [batch_size, sequence_length]
        # y: [batch_size, sequence_length + 1] (1 padding + sequence length)
        batch_size = x.shape[0]
        sequence_length = y.shape[1]
        
        outputs = torch.zeros(sequence_length, batch_size, self.num_vocab).to(self.device)
        
        encoder_output, (h_t, c_t) = self.encoder(x)
        
        y = y.transpose(0, 1)
        # y: [sequence_length + 1, batch_size]
        cur_y = y[0]
        # cur_y: [1, batch_size]
        
        for target in range(1, sequence_length):
            output, (h_t, c_t), attention_weights = self.decoder(cur_y, encoder_output, h_t, c_t)
            
            outputs[target] = output
            
            best_guess = output.argmax(1)
            # best_guess: [batch_size, num_vocab]
            
            cur_y = y[target] if random.random() < self.teacher_force_ratio else best_guess
        
        return outputs, attention_weights