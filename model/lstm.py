import numpy as np
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import sys
sys.path.insert(0,"./")
from utils.parser import args
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5"
class lstm_encoder(nn.Module):
    ''' Encodes time-series sequence '''

    def __init__(self, input_size, hidden_size, num_layers = 1):
        
        super(lstm_encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # define LSTM layer
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size,
                            num_layers = num_layers)

    def forward(self, x_input):

        
        lstm_out, self.hidden = self.lstm(x_input.view(x_input.shape[0], x_input.shape[1], self.input_size))
        
        return lstm_out, self.hidden     
    
    def init_hidden(self, batch_size):
        
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))


class lstm_decoder(nn.Module):
    ''' Decodes hidden state output by encoder '''
    
    def __init__(self, input_size, hidden_size,output_size, num_layers = 1):

        '''
        : param input_size:     the number of features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        '''
        
        super(lstm_decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size,
                            num_layers = num_layers)
        self.linear = nn.Linear(hidden_size, output_size)           

    def forward(self, x_input, encoder_hidden_states):
        
        lstm_out, self.hidden = self.lstm(x_input.unsqueeze(0), encoder_hidden_states)
        output = self.linear(lstm_out.squeeze(0))     
        
        return output, self.hidden

class lstm_seq2seq(nn.Module):
    ''' train LSTM encoder-decoder and make predictions '''
    
    def __init__(self, opt,hidden_dim=16):

        '''
        : param input_size:     the number of expected features in the input X
        : param hidden_size:    the number of features in the hidden state h
        '''

        super(lstm_seq2seq, self).__init__()

        self.input_dim = opt.input_dim
        self.hidden_dim = hidden_dim
        self.input_n = opt.input_n
        self.output_n = 50
        
        self.embed_dim = 16
        self.output_dim = 2

        # self.embedding = nn.ModuleList()
        # for i in range(self.input_dim):
        #     emb = nn.Linear(1, self.embed_dim)
        #     self.embedding.append(emb)
        self.embed_layer = nn.Linear(self.input_dim,self.embed_dim)
        
        self.encoder = lstm_encoder(input_size = self.embed_dim, hidden_size = hidden_dim)
        self.decoder = lstm_decoder(input_size = self.embed_dim, hidden_size = hidden_dim,output_size=self.output_dim)
        
        
    def forward(self, input):
        # outputs tensor
        batch_size,in_n,c =input.shape
        input = input.transpose(0,1)
        present = input[-1].clone()

        outputs = torch.zeros(self.output_n, batch_size,self.output_dim).cuda()  #grad = True
        encoder_hidden = self.encoder.init_hidden(batch_size)

        # encoder outputs
        encoder_output, encoder_hidden = self.encoder(self.embed_layer(input))

        # decoder with teacher forcing
        decoder_input = input[-1, :, :]   # shape: (batch_size, input_size)   #grad = True
        decoder_hidden = encoder_hidden

        # predict recursively
        for t in range(self.output_n): 
            decoder_output, decoder_hidden = self.decoder(self.embed_layer(decoder_input), decoder_hidden)
            decoder_input = decoder_output
            outputs[t] = decoder_output
        # outputs = outputs.transpose(0,1) + present.unsqueeze(1).expand(-1,self.output_n,-1)
        outputs = outputs.transpose(0,1)
        return outputs
