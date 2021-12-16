import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 0
EOS_token = 1
hidden_size = 256
latent_size = 32
embedding_size = 8
vocab_size = 28
MAX_LENGTH = 25

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)

        self.h_mu = nn.Linear(hidden_size, latent_size)
        self.h_logvar = nn.Linear(hidden_size, latent_size)
        self.c_mu = nn.Linear(hidden_size, latent_size)
        self.c_logvar = nn.Linear(hidden_size, latent_size)

    def forward(self, input, hidden, cell):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, (hidden, cell) = self.lstm(output, (hidden, cell))
        return output, hidden, cell, self.h_mu(hidden), self.h_logvar(hidden), self.c_mu(cell), self.c_logvar(cell)

    def initHidden(self, data_c):
        hidden = torch.zeros(1, 1, self.hidden_size-data_c.size(2), device=device)
        hidden = torch.cat((hidden, data_c), 2) 
        return hidden


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden, cell):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, (hidden, cell) = self.lstm(output, (hidden, cell))
        output = self.out(output[0])
        return output, hidden, cell

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class VAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, output_size):
        super(VAE, self).__init__()

        self.embedding = nn.Embedding(5, embedding_size)
        self.encoder = EncoderRNN(input_size, hidden_size, latent_size)
        self.fc1 = nn.Linear(latent_size+embedding_size, hidden_size)
        self.fc2 = nn.Linear(latent_size+embedding_size, hidden_size)
        self.decoder = DecoderRNN(hidden_size, output_size)
    
    def forward(self, data, data_c, target_c):
        data_c = self.embedding(data_c)
        
        encoder_hidden = self.encoder.initHidden(data_c)
        encoder_cell = self.encoder.initHidden(data_c)

        for ei in range(data.size(0)):
            encoder_output, encoder_hidden, encoder_cell, h_mu, h_logvar, c_mu, c_logvar = \
                self.encoder(data[ei], encoder_hidden, encoder_cell)
        
        return self.generate(h_mu, c_mu, target_c)

    def generate(self, h_latent, c_latent, target_c):
        decoder_input = torch.tensor([[SOS_token]], device=device)
        target_c = self.embedding(target_c)
        decoder_hidden = torch.cat((h_latent, target_c), 2)
        decoder_hidden = self.fc1(decoder_hidden)
        
        decoder_cell = torch.cat((c_latent, target_c), 2)
        decoder_cell = self.fc2(decoder_cell)

        pred_list = []
        for di in range(MAX_LENGTH):
            decoder_output, decoder_hidden, decoder_cell = \
                self.decoder(decoder_input, decoder_hidden, decoder_cell)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            pred_list.append(idx2chr(topi))
            if decoder_input.item() == EOS_token:
                break
        
        pred = ''.join(pred_list)
        return pred

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

def idx2chr(index):
    if index >= 2:
        return chr(index-2+ord('a'))
    else:
        return ''