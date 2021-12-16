import torch
from torch.utils import data
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torchvision
import time
import numpy as np
from sklearn.metrics import confusion_matrix

from dataloader import *
from model import *

def train(epoch):
    model.train()
    correct = 0
    for idx, (x_batch, y_batch) in enumerate(train_loader):
        optimizer.zero_grad()
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        output = model(x_batch)
        y_batch = y_batch.long()
        loss = criterion(output, y_batch)

        loss.backward()
        optimizer.step()

        pred = torch.max(output.data, 1)[1].cuda()
        correct += pred.eq(y_batch.data).sum()

        if idx % 250 == 0:
                print('Train Epoch: {} [{}/{}]\t'.format(
                    epoch, idx * len(x_batch), len(train_loader.dataset)))
    
    acc = correct / float(len(train_loader.dataset))
    print('Train Epoch: {} Accuracy: {:.6f}'.format(epoch, acc))

    return acc

def test(epoch):
    model.eval()
    correct = 0
    pred_list = []
    true_list = []
    for idx, (x_batch, y_batch) in enumerate(test_loader):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        output = model(x_batch)
        y_batch = y_batch.long()
        
        pred = torch.max(output.data, 1)[1].cuda()
        correct += pred.eq(y_batch.data).sum()

        pred_list.extend(pred.cpu().numpy())
        true_list.extend(y_batch.cpu().numpy())
    
    acc = correct / float(len(test_loader.dataset))
    print('Test Epoch: {} Accuracy: {:.6f}'.format(epoch, acc))

    return acc, pred_list, true_list

def plot(model_name):
    plt.clf()
    plt.title('{} Comparision'.format(model_name))
    plt.plot(axis, train_acc_list, color='tab:orange', label = 'Train (w/o pretraining)')
    plt.plot(axis, test_acc_list, color='tab:blue', label = 'Test (w/o pretraining)')
    plt.plot(axis, pre_train_acc_list, color='tab:green', label = 'Train (with pretraining)')
    plt.plot(axis, pre_test_acc_list, color='tab:red', label = 'Test (with pretraining)')
    plt.legend(loc = 'upper left')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy(%)')
    plt.xlim((1-0.1, epochs+0.1))
    plt.ylim((60, 90))
    plt.xticks(np.arange(1, epochs+0.5, 1))
    plt.yticks(np.arange(60, 90+1, 5))
    plt.savefig('{}_Accuracy.jpg'.format(model_name))

def plot_confusion_matrix(y_true, y_pred, model_name, classes=5, cmap=plt.cm.Blues):
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.clf()
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
            xticklabels=[0, 1, 2, 3, 4], yticklabels=[0, 1, 2, 3, 4],
            title='Normalized confusion matrix',
            ylabel='True label', xlabel='Predicted label')
        
           
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig('{}_confusion.png'.format(model_name))


if __name__ == '__main__':
    batch_size = 4

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = RetinopathyLoader('./data/', 'train')
    test_dataset  = RetinopathyLoader('./data/', 'test')
    train_loader = data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
    test_loader  = data.DataLoader(dataset = test_dataset,  batch_size = batch_size, shuffle = True)
    
    ###### ResNet18 without pretraining ######
    epochs = 10
    axis = range(1, epochs+1)
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)

    train_acc_list = []
    test_acc_list = []
    max_acc = 0

    max_pred_list = []
    max_true_list = []
    
    t0 = time.time()
    for epoch in range(epochs):
        acc_train = train(epoch)
        acc_test, pred_list, true_list = test(epoch)
        train_acc_list.append(acc_train.item() * 100)
        test_acc_list.append(acc_test.item() * 100)
        if acc_test > max_acc:
            torch.save(model.state_dict(), 'resnet18.pkl')
            max_acc = acc_test
            max_pred_list = pred_list
            max_true_list = true_list
        
    t1 = time.time()
    time_elapsed = t1 - t0
    print('ResNet18 time {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('ResNet18 without Pretraining Testing Accuracy: {}'.format(max_acc))

    plot_confusion_matrix(max_true_list, max_pred_list, 'resnet18')

    ###### ResNet18 with pretraining ######
    epochs = 10
    axis = range(1, epochs+1)
    model = torchvision.models.resnet18(pretrained=True)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    model.fc = nn.Linear(512, 5, bias=True)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4, nesterov=True)

    pre_train_acc_list = []
    pre_test_acc_list = []
    max_acc = 0
    
    max_pred_list = []
    max_true_list = []

    t2 = time.time()
    for epoch in range(epochs):
        acc_train = train(epoch)
        acc_test, pred_list, true_list = test(epoch)
        pre_train_acc_list.append(acc_train.item() * 100)
        pre_test_acc_list.append(acc_test.item() * 100)
        if acc_test > max_acc:
            torch.save(model.state_dict(), 'resnet18_pre.pkl')
            max_acc = acc_test
            max_pred_list = pred_list
            max_true_list = true_list
    
    t3 = time.time()
    time_elapsed = t3 - t2
    print('ResNet18 time {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('ResNet18 with Pretraining Testing Accuracy: {}'.format(max_acc))

    plot_confusion_matrix(max_true_list, max_pred_list, 'resnet18_pre')

    plot('ResNet18')

    ###### ResNet50 without pretraining ######
    epochs = 5
    axis = range(1, epochs+1)
    model = ResNet(Bottlenet, [3, 4, 6, 3])
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)

    train_acc_list = []
    test_acc_list = []
    max_acc = 0
    max_pred_list = []
    max_true_list = []

    t4 = time.time()
    for epoch in range(epochs):
        acc_train = train(epoch)
        acc_test, pred_list, true_list = test(epoch)
        train_acc_list.append(acc_train.item() * 100)
        test_acc_list.append(acc_test.item() * 100)
        if acc_test > max_acc:
            torch.save(model.state_dict(), 'resnet50.pkl')
            max_acc = acc_test
            max_pred_list = pred_list
            max_true_list = true_list

    t5 = time.time()
    time_elapsed = t5 - t4
    print('ResNet50 time {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('ResNet50 without Pretraining Testing Accuracy: {}'.format(max_acc))

    plot_confusion_matrix(max_true_list, max_pred_list, 'resnet50')

    ###### ResNet50 with pretraining ######
    epochs = 5
    axis = range(1, epochs+1)
    model = torchvision.models.resnet50(pretrained=True)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    model.fc = nn.Linear(512*4, 5, bias=True)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)

    pre_train_acc_list = []
    pre_test_acc_list = []
    max_acc = 0

    max_pred_list = []
    max_true_list = []

    t6 = time.time()
    for epoch in range(epochs):
        acc_train = train(epoch)
        acc_test, pred_list, true_list = test(epoch)
        pre_train_acc_list.append(acc_train.item() * 100)
        pre_test_acc_list.append(acc_test.item() * 100)
        if acc_test > max_acc:
            torch.save(model.state_dict(), 'resnet50_pre.pkl')
            max_acc = acc_test
            max_pred_list = pred_list
            max_true_list = true_list

    t7 = time.time()
    time_elapsed = t7 - t6
    print('ResNet50 time {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('ResNet50 with Pretraining Testing Accuracy: {}'.format(max_acc))

    plot_confusion_matrix(max_true_list, max_pred_list, 'resnet50_pre')

    plot('ResNet50')
    
    








    