import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt

from dataloader import read_bci_data
from model import *
import matplotlib
matplotlib.use('Agg')

def train(epoch):
    model.train()
    correct = 0
    for idx, (x_batch, y_batch) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(x_batch)
        y_batch = y_batch.long()
        loss = Loss(output, y_batch)

        loss.backward()
        optimizer.step()

        _, pred = torch.max(output.data, 1)
        correct += pred.eq(y_batch.data).sum()   

    acc = correct / float(len(train_loader.dataset))
    print('Train Epoch: {} Accuracy: {:.6f}'.format(epoch, acc))

    return acc

def test(epoch):
    model.eval()
    correct = 0
    test_loss = 0
    global max_acc
    for x_batch, y_batch in test_loader:
        output = model(x_batch)
        y_batch = y_batch.long()
        test_loss += Loss(output, y_batch)

        _, pred = torch.max(output.data, 1)
        correct += pred.eq(y_batch.data).sum()

    acc = correct / float(len(test_loader.dataset))
    max_acc = max(acc, max_acc)
    print('Test Epoch: {} Accuracy: {:.6f}'.format(epoch, acc))

    return acc

def prepare_data():
    x_train, y_train, x_test, y_test = read_bci_data()

    x_train = torch.Tensor(x_train)
    y_train = torch.Tensor(y_train)
    x_test  = torch.Tensor(x_test)
    y_test  = torch.Tensor(y_test)

    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_test  = x_test.to(device)
    y_test  = y_test.to(device)

    train_dataset = Data.TensorDataset(x_train, y_train)
    train_loader = Data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
    test_dataset = Data.TensorDataset(x_test, y_test)
    test_loader = Data.DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = True)

    return train_loader, test_loader

def plot(model_name):
    plt.clf()
    plt.title('Activation function comparision({})'.format(model_name))
    plt.plot(axis, relu_train_acc, color='tab:orange', label = 'relu_train')
    plt.plot(axis, relu_test_acc, color='tab:blue', label = 'relu_test')
    plt.plot(axis, leaky_relu_train_acc, color='tab:green', label = 'leaky_relu_train')
    plt.plot(axis, leaky_relu_test_acc, color='tab:red', label = 'leaky_relu_test')
    plt.plot(axis, elu_train_acc, color='tab:purple', label = 'elu_train')
    plt.plot(axis, elu_test_acc, color='tab:brown', label = 'elu_test')
    plt.legend(loc = 'lower right')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy(%)')
    plt.xlim((0, epochs))
    plt.ylim((65, 100))
    plt.savefig('{}_Accuracy.jpg'.format(model_name))
    #plt.show()
    

if __name__ == '__main__':
    epochs = 300
    axis = range(epochs)
    batch_size = 108
    lr = 0.001
    total_max_acc = 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_loader, test_loader = prepare_data()
    
    ###### EEGNET ELU ######
    model = EEGNET_ELU()
    model.to(device)

    Loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    max_acc = 0
    elu_train_acc = []
    elu_test_acc  = []

    for epoch in range(epochs):
        acc_train = train(epoch)
        acc_test  = test(epoch)
        if acc_test > total_max_acc:
            torch.save(model.state_dict(), 'net_params.pkl')
            m_name = 'EEGNET_ELU'
            total_max_acc = acc_test

        elu_train_acc.append(acc_train * 100)
        elu_test_acc.append(acc_test * 100)
    
    print('EEGNET ELU Testing Accuracy: {:.6f}\n'.format(max_acc.item()))

    ###### EEGNET RELU ######
    model = EEGNET_RELU()
    model.to(device)

    Loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    max_acc = 0
    relu_train_acc = []
    relu_test_acc  = []

    for epoch in range(epochs):
        acc_train = train(epoch)
        acc_test  = test(epoch)
        if acc_test > total_max_acc:
            torch.save(model.state_dict(), 'net_params.pkl')
            m_name = 'EEGNET_RELU'
            total_max_acc = acc_test

        relu_train_acc.append(acc_train * 100)
        relu_test_acc.append(acc_test * 100)
    
    print('EEGNET RELU Testing Accuracy: {:.6f}\n'.format(max_acc.item()))

    ###### EEGNET LEAKY RELU ######
    model = EEGNET_LEAKY_RELU()
    model.to(device)

    Loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    max_acc = 0
    leaky_relu_train_acc = []
    leaky_relu_test_acc  = []

    for epoch in range(epochs):
        acc_train = train(epoch)
        acc_test  = test(epoch)
        if acc_test > total_max_acc:
            torch.save(model.state_dict(), 'net_params.pkl')
            m_name = 'EEGNET_LEAKY_RELU'
            total_max_acc = acc_test

        leaky_relu_train_acc.append(acc_train * 100)
        leaky_relu_test_acc.append(acc_test * 100)
    
    print('EEGNET LEAKY RELU Testing Accuracy: {:.6f}\n'.format(max_acc.item()))

    plot('EEGNET')



    ###### DEEPCONVNET ELU ######
    model = DeepConvNet_ELU()
    model.to(device)

    Loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    max_acc = 0
    elu_train_acc = []
    elu_test_acc  = []

    for epoch in range(epochs):
        acc_train = train(epoch)
        acc_test  = test(epoch)
        if acc_test > total_max_acc:
            torch.save(model.state_dict(), 'net_params.pkl')
            m_name = 'DEEP_ELU'
            total_max_acc = acc_test

        elu_train_acc.append(acc_train * 100)
        elu_test_acc.append(acc_test * 100)
    
    print('DeepConvNet ELU Testing Accuracy: {:.6f}\n'.format(max_acc.item()))

    ###### DEEPCONVNET RELU ######
    model = DeepConvNet_RELU()
    model.to(device)

    Loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    max_acc = 0
    relu_train_acc = []
    relu_test_acc  = []

    for epoch in range(epochs):
        acc_train = train(epoch)
        acc_test  = test(epoch)
        if acc_test > total_max_acc:
            torch.save(model.state_dict(), 'net_params.pkl')
            m_name = 'DEEP_RELU'
            total_max_acc = acc_test

        relu_train_acc.append(acc_train * 100)
        relu_test_acc.append(acc_test * 100)
    
    print('DeepConvNet RELU Testing Accuracy: {:.6f}\n'.format(max_acc.item()))

    ###### DEEPCONVNET LEAKY RELU ######
    model = DeepConvNet_LEAKY_RELU()
    model.to(device)

    Loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    max_acc = 0
    leaky_relu_train_acc = []
    leaky_relu_test_acc  = []

    for epoch in range(epochs):
        acc_train = train(epoch)
        acc_test  = test(epoch)
        if acc_test > total_max_acc:
            torch.save(model.state_dict(), 'net_params.pkl')
            m_name = 'DEEP_LEAKY_RELU'
            total_max_acc = acc_test

        leaky_relu_train_acc.append(acc_train * 100)
        leaky_relu_test_acc.append(acc_test * 100)
    
    print('DeepConvNet Leaky RELU Testing Accuracy: {:.6f}\n'.format(max_acc.item()))

    plot('DeepConvNet')

    print('max total acc: ', total_max_acc.item())
    print('model name: ', m_name)


