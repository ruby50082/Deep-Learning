from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import time
import math
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np
from os import system
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from model import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 0
EOS_token = 1
#----------Hyper Parameters----------#
hidden_size = 256
latent_size = 32
embedding_size = 8
vocab_size = 28
teacher_forcing_ratio = 0.0
kld_weight = 0.0
LR = 0.01
isMonotonic = True

MAX_LENGTH = 25
tense2idx = {'sp': 0, 'tp': 1, 'pg': 2, 'p': 3}

def TrainData():
    data = []
    with open('./data/train.txt', 'r') as f:
        for line in f.readlines():
            for idx, word in enumerate(line.strip('\n').split(' ')):
                data.append(((word, idx), (word, idx)))
    return data

def TestData():
    data = []
    with open('./data/test.txt', 'r') as f1, open('./data/test_tense.txt', 'r') as f2:
        for line1, line2 in zip(f1.readlines(), f2.readlines()):
            word  = line1.strip('\n').split(' ')
            tense = line2.strip('\n').split(' ')
            data.append(((word[0], tense2idx[tense[0]]), (word[1], tense2idx[tense[1]])))
    return data

def word2num(word):
    indeaxis = [ord(w)-ord('a')+2 for w in word]
    indeaxis.append(EOS_token)
    return torch.tensor(indeaxis, dtype=torch.long, device=device).view(-1, 1)

def pair2num(pair):
    data = word2num(pair[0][0])
    target = word2num(pair[1][0])
    data_c = torch.tensor(pair[0][1], dtype=torch.long, device=device).view(1, -1)
    target_c = torch.tensor(pair[1][1], dtype=torch.long, device=device).view(1, -1)
    return (data, target), (data_c, target_c)


def compute_bleu(output, reference): # (word1, word2)
    cc = SmoothingFunction()
    if len(reference) == 3:
        weights = (0.33,0.33,0.33)
    else:
        weights = (0.25,0.25,0.25,0.25)
    return sentence_bleu([reference], output,weights=weights,smoothing_function=cc.method1)

def Gaussian_score(words):
    words_list = []
    score = 0
    yourpath = './data/train.txt' # should be your directory of train.txt
    with open(yourpath,'r') as fp:
        for line in fp:
            word = line.split(' ')
            word[3] = word[3].strip('\n')
            words_list.extend([word])
        for t in words:
            for i in words_list:
                if t == i:
                    score += 1
    return score/len(words)

def loss_function(y_pred, y, h_mu, h_logvar, c_mu, c_logvar):
    cross_entropy = F.cross_entropy(y_pred, y)
    h_kld = -0.5 * torch.sum(1 + h_logvar - h_mu.pow(2) - h_logvar.exp())
    c_kld = -0.5 * torch.sum(1 + c_logvar - c_mu.pow(2) - c_logvar.exp())
    return cross_entropy, h_kld, c_kld

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def KL_annealing(iter, isMonotonic): # kld_weight
    if isMonotonic:
        return min(1.0, iter/50000)
    else:
        return min(1.0, (iter % 20000)/10000)

def plot(n_iters, plot_every, cross_loss_list, kld_loss_list, bleu_list, gaussian_list):
    axis = [i for i in range(1, n_iters+1, plot_every)]
    
    plt.clf()
    plt.title('Loss')
    plt.plot(axis, cross_loss_list, color='tab:orange', label='CrossEntropy')
    plt.plot(axis, kld_loss_list, color='tab:blue', label='KL loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('./Loss.png')
    
    plt.clf()
    plt.title('BLEU Score')
    plt.plot(axis, bleu_list, color='tab:purple', label='BLEU score')
    plt.xlabel('Iterations')
    plt.ylabel('BLEU Score')
    plt.legend()
    plt.savefig('./BLEU_Score.png')
    
    plt.clf()
    plt.title('Gaussian Score')
    plt.plot(axis, gaussian_list, color='tab:brown', label='Gaussian score')
    plt.xlabel('Iterations')
    plt.ylabel('Gaussian Score')
    plt.legend()
    plt.savefig('./Gaussian_Score.png')


def train(data, target, data_c, target_c, model, optimizer, criterion, kld_weight, max_length=MAX_LENGTH):
    model.train()
    data_c = model.embedding(data_c)
    encoder_hidden = model.encoder.initHidden(data_c)
    encoder_cell = model.encoder.initHidden(data_c)

    optimizer.zero_grad()

    input_length = data.size(0)
    target_length = target.size(0)

    cross_loss = 0
    kld_loss = 0

    #----------sequence to sequence part for encoder----------#
    for ei in range(input_length):
        encoder_output, encoder_hidden, encoder_cell, h_mu, h_logvar, c_mu, c_logvar = \
            model.encoder(data[ei], encoder_hidden, encoder_cell)

    decoder_input = torch.tensor([[SOS_token]], device=device)

    target_c = model.embedding(target_c)
    h_latent = model.reparameterize(h_mu, h_logvar)
    decoder_hidden = torch.cat((h_latent, target_c), 2)
    decoder_hidden = model.fc1(decoder_hidden)
    
    c_latent = model.reparameterize(c_mu, c_logvar)
    decoder_cell = torch.cat((c_latent, target_c), 2)
    decoder_cell = model.fc2(decoder_cell)


    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
	

    #----------sequence to sequence part for decoder----------#
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_cell = \
                model.decoder(decoder_input, decoder_hidden, decoder_cell)
            _cross_loss, h_kld_loss, c_kld_loss = \
                criterion(decoder_output, target[di], h_mu, h_logvar, c_mu, c_logvar)
            
            kld_loss = (h_kld_loss+c_kld_loss)/2
            cross_loss += _cross_loss
            decoder_input = target[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_cell = \
                model.decoder(decoder_input, decoder_hidden, decoder_cell)
            
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            _cross_loss, h_kld_loss, c_kld_loss = \
                criterion(decoder_output, target[di], h_mu, h_logvar, c_mu, c_logvar)
            
            kld_loss = (h_kld_loss+c_kld_loss)/2
            cross_loss += _cross_loss
            if decoder_input.item() == EOS_token:
                break

    loss = cross_loss + kld_weight * kld_loss
    loss.backward()

    optimizer.step()

    return loss.item()/target_length, cross_loss.item()/target_length, kld_loss/target_length


def evaluate(model, isPrint):
    model.eval()
    with torch.no_grad():
        testing_pairs = []
        testing_cs = []
        for i in range(len(test_data)):
            pairs = pair2num(test_data[i])
            testing_pairs.append(pairs[0])
            testing_cs.append(pairs[1])
        
        bleu_total = 0
        for i in range(len(test_data)):
            x = test_data[i][0][0]
            target = test_data[i][1][0]
            y_pred = model(testing_pairs[i][0], testing_cs[i][0], testing_cs[i][1])
            bleu_total += compute_bleu(y_pred, target)
            
            if isPrint:
                print('\ninput: {}'.format(x))
                print('target: {}'.format(target))
                print('pred: {}'.format(y_pred))
        
        if isPrint:
            print('\nAverage BLEU-4 score: {}'.format(bleu_total/len(test_data)))

        #### gaussian score ####
        words_list = []
        for i in range(100):
            h_latent = torch.randn(1,1,latent_size, device=device)
            c_latent = torch.randn(1,1,latent_size, device=device)
            word = []
            for j in range(4):
                word.append(model.generate(h_latent, c_latent, torch.tensor(j, dtype=torch.long, device=device).view(1, -1)))
            words_list.append(word)
        
        gaussian_score = Gaussian_score(words_list)
        if isPrint:
            for i, word in enumerate(words_list):
                print(word)
            print('Gaussian score: {}'.format(gaussian_score))
    
    return bleu_total/len(test_data), gaussian_score

def trainIters(model, n_iters, print_every=1000, plot_every=100, learning_rate=0.05):
    start = time.time()
    cross_loss_list, kld_loss_list, bleu_list, gaussian_list = [], [], [], []
    print_loss_total = 0  # Reset every print_every
    plot_cross_total = 0  # Reset every plot_every
    plot_kld_total = 0

    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    training_pairs = []
    training_cs = []
    for i in range(n_iters):
        pairs = pair2num(random.choice(train_data))
        training_pairs.append(pairs[0])
        training_cs.append(pairs[1])

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        training_c = training_cs[iter - 1]
        data = training_pair[0]
        target = training_pair[1]
        data_c = training_c[0]
        target_c = training_c[1]

        loss, cross_loss, kld_loss = train(data, target, data_c, target_c,
                     model, optimizer, loss_function, KL_annealing(iter, isMonotonic))
        print_loss_total += loss
        plot_cross_total += cross_loss
        plot_kld_total += kld_loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))
        if iter % plot_every == 0:
            plot_cross_avg = plot_cross_total / plot_every
            plot_cross_total = 0
            plot_kld_avg = plot_kld_total / plot_every
            plot_kld_total = 0
            cross_loss_list.append(plot_cross_avg)
            kld_loss_list.append(plot_kld_avg)
            if iter % n_iters == 0:
                bleu, gaussian_score = evaluate(model, True)
            else:
                bleu, gaussian_score = evaluate(model, False)
            bleu_list.append(bleu)
            gaussian_list.append(gaussian_score)
            if bleu >= 0.7 and gaussian_score >= 0.3:
                torch.save(model.state_dict(), f'./model/model_{bleu:.2f}_{gaussian_score}.pkl')
                evaluate(model, True)
                print('Iter {}: bleu: {}, gaussian: {}'.format(i, bleu, gaussian_score))

    plot(n_iters, plot_every, cross_loss_list, kld_loss_list, bleu_list, gaussian_list)



if __name__ == '__main__':
    train_data = TrainData()
    test_data = TestData()
    model = VAE(vocab_size, hidden_size, latent_size, vocab_size).to(device)
    trainIters(model, 100000, print_every=5000, plot_every=100, learning_rate=LR)