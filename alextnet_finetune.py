from __future__ import print_function

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import warnings
warnings.filterwarnings('ignore')

import math
import model
import torch
import dataloader

from torch import nn
from torch import optim
from torch.autograd import Variable

cuda = torch.cuda.is_available()

def step_decay(epoch, learning_rate):
    """
    learning rate step decay
    :param epoch: current training epoch
    :param learning_rate: initial learning rate
    :return: learning rate after step decay
    """
    initial_lrate = learning_rate
    drop = 0.8
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate

def train_alexnet(epoch, model, learning_rate, source_loader):
    """
    train source on alexnet
    :param epoch: current training epoch
    :param model: defined alexnet
    :param learning_rate: initial learning rate
    :param source_loader: source loader
    :return:
    """
    log_interval = 10
    LEARNING_RATE = step_decay(epoch, learning_rate)
    print(f'Learning Rate: {LEARNING_RATE}')
    optimizer = optim.Adam([
        {'params': model.features.parameters()},
        {'params': model.classifier.parameters()},
        {'params': model.final_classifier.parameters(), 'lr': LEARNING_RATE}
    ], lr=LEARNING_RATE / 10)

    # enter training mode
    model.train()

    iter_source = iter(source_loader)
    num_iter = len(source_loader)

    correct = 0
    total_loss = 0
    clf_criterion = nn.CrossEntropyLoss()

    for i in range(1, num_iter):
        source_data, source_label = iter_source.next()
        if cuda:
            source_data, source_label = source_data.cuda(), source_label.cuda()
        source_data, source_label = Variable(source_data), Variable(source_label)

        optimizer.zero_grad()

        source_preds = model(source_data)
        preds = source_preds.data.max(1, keepdim=True)[1]
        correct += preds.eq(source_label.data.view_as(preds)).sum()

        loss = clf_criterion(source_preds, source_label)
        total_loss += loss

        loss.backward()
        optimizer.step()

        if i % log_interval == 0:
            print('Train Epoch {}: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, i * len(source_data), len(source_loader) * BATCH_SIZE,
                100. * i / len(source_loader), loss.data[0]))

    total_loss /= len(source_loader)
    acc_train = float(correct) * 100. / (len(source_loader) * BATCH_SIZE)

    print('{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        SOURCE_NAME, total_loss.data[0], correct, len(source_loader.dataset), acc_train))


def test_alexnet(model, target_loader):
    """
    test target data on fine-tuned alexnet
    :param model: trained alexnet on source data set
    :param target_loader: target dataloader
    :return: correct num
    """
    # enter evaluation mode
    clf_criterion = nn.CrossEntropyLoss()

    model.eval()
    test_loss = 0
    correct = 0

    for data, target in target_test_loader:
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        target_preds = model(data)
        test_loss += clf_criterion(target_preds, target) # sum up batch loss
        pred = target_preds.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(target_loader)
    print('{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        TARGET_NAME, test_loss.data[0], correct, len(target_loader.dataset),
        100. * correct / len(target_loader.dataset)))
    return correct


if __name__ == '__main__':

    ROOT_PATH = '../../data/Office31'
    SOURCE_NAME = 'amazon'
    TARGET_NAME = 'webcam'

    BATCH_SIZE = 256
    TRAIN_EPOCHS = 200
    learning_rate = 1e-3

    source_loader = dataloader.load_training(ROOT_PATH, SOURCE_NAME, BATCH_SIZE)
    target_train_loader = dataloader.load_training(ROOT_PATH, TARGET_NAME, BATCH_SIZE)
    target_test_loader = dataloader.load_testing(ROOT_PATH, TARGET_NAME, BATCH_SIZE)
    print('Load data complete')

    alexnet = model.Alexnet_finetune(num_classes=31)
    print('Construct model complete')

    # load pretrained alexnet model
    alexnet = model.load_pretrained_alexnet(alexnet)
    print('Load pretrained alexnet parameters complete\n')

    if cuda: alexnet.cuda()

    for epoch in range(1, TRAIN_EPOCHS + 1):
        print(f'Train Epoch {epoch}:')
        train_alexnet(epoch, alexnet, learning_rate, source_loader)
        correct = test_alexnet(alexnet, target_test_loader)
