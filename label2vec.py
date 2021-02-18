from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torchvision import models
import torch
from torch import nn
from src.model import Classifier, BYOL_Pre
from src.utils import get_file
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torchvision import transforms as T
from src.resnet import resnet18
from argparse import ArgumentParser


transform = T.Compose(
            [
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.RandomGrayscale(),
                T.RandomApply([T.ColorJitter()]),
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
            ]
        )
TRAIN_DATASET = CIFAR10(root="data", train=True, download=False, transform=transform)
TEST_DATASET = CIFAR10(root="data", train=False, download=False, transform=transform)

sim = nn.CosineSimilarity()
def ours_loss(temp):
    sim_ = nn.CosineSimilarity(2)
    sim_f = nn.CosineSimilarity(1)
    def criterion_(feature_batch, label_batch, vectors):
        batch_size, feature_size = feature_batch.shape
        tiled_f = feature_batch.repeat(1,10).view(batch_size, -1, feature_size) # 256 * 10 * 512
        tiled_l = vectors[label_batch].repeat(1,10).view(batch_size, -1, feature_size) # 256 * 10 * 512
        class_num, feature_size = vectors.shape
        tiled_v = vectors.repeat(batch_size, 1).view(batch_size, class_num, feature_size) # 256 * 10 * 512
        tiled_sim_f = sim_(tiled_f, tiled_v) # 256 * 10
        tiled_sim_l = sim_(tiled_l, tiled_v) # 256 * 10
        e_sims_f = torch.exp(tiled_sim_f/temp) # 256 * 10
        e_sims_l = torch.exp(tiled_sim_l/temp) # 256 * 10
        e_sum_f = torch.sum(e_sims_f, dim=1).unsqueeze(0).transpose(0,1) # 256 * 1
        e_sum_l = torch.sum(e_sims_l, dim=1).unsqueeze(0).transpose(0,1) # 256 * 1
        loss = 1-torch.mean(sim_f(torch.log(e_sims_f/e_sum_f), torch.log(e_sims_l/e_sum_l)))
        return loss
    return criterion_
         
def contrastive_loss(temp):
    sim_ = nn.CosineSimilarity(2)
    def criterion_(feature_batch, label_batch, vectors):
        batch_size, feature_size = feature_batch.shape
        tiled_f = feature_batch.repeat(1,10).view(batch_size, -1, feature_size) # 256 * 10 * 512
        class_num, feature_size = vectors.shape
        tiled_v = vectors.repeat(batch_size, 1).view(batch_size, class_num, feature_size)
        tiled_sim = sim_(tiled_f, tiled_v)
        e_sims = torch.exp(tiled_sim/temp) # 256 * 10
        e_lab = torch.gather(e_sims.transpose(0,1), 0, label_batch.unsqueeze(0))[0]
        e_sum = torch.sum(e_sims, dim=1)
        loss = torch.mean(-torch.log(e_lab/e_sum))
        return loss
    return criterion_
criterion_c = contrastive_loss(0.07)
criterion = ours_loss(0.07)


class NewCIFAR10(Dataset):
    def __init__(self, data, labels):
        self.data=data
        self.labels=labels
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.data[idx], self.labels[idx]

import torch.nn as nn

class Label2Vec(nn.Module):
    def __init__(self):
        super().__init__()
        self.l2v = nn.Embedding(10,512)
    def forward(self, label):
        return self.l2v(label)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

device = torch.device('cuda:0')
l2v = Label2Vec()
l2v.to(device)

if __name__=='__main__':
    # model = models.resnet18(pretrained=False).to(device)
    model = resnet18()['backbone'].to(device)
    # pre_file = get_file('version_0OOFJ_finetuned_adam.ckpt')
    # if pre_file is not None:
    #         cls = Classifier.load_from_checkpoint(pre_file, model=model)
    #         model.load_state_dict(cls.model.state_dict())
    model.fc = Identity()

    lr = 0.01

    # optimizer = optim.Adam(list(model.parameters()) + list(l2v.parameters()), lr=0.0001) 
    # optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01)
    train_loader = DataLoader(TRAIN_DATASET, batch_size=512, shuffle=True)
    test_loader = DataLoader(TEST_DATASET, batch_size=5000, shuffle=True)
    epochs = 500
    sim_ = nn.CosineSimilarity(2)
    wd = 0.1
    for epoch in range(epochs):
        if (epoch+1)%20==0:
            wd *= 0.8
        # optimizer1 = optim.Adam(model.parameters(), lr=lr)
        optimizer1 = optim.SGD(model.parameters(), lr=0.01, weight_decay=wd)#, momentum=0.9)
        optimizer2 = optim.SGD(l2v.parameters(), lr=0.01, weight_decay=wd)#, momentum=0.9)
        # optimizer2 = optim.Adam(l2v.parameters(), lr=lr*5)
        # lr = lr * 0.95
        for xi, x in enumerate(train_loader):
            dat, lab = x
            dat = dat.to(device)
            lab = lab.to(device)
            
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            new_vector = l2v(torch.LongTensor([0,1,2,3,4,5,6,7,8,9]).to(device))
            lab_hat = model(dat)
            loss1 = criterion_c(lab_hat, lab, new_vector)
            loss2 = 10*criterion(lab_hat, lab, new_vector)
            loss3 = 1-torch.mean(sim(lab_hat, new_vector[lab]))
            loss = loss1 + loss2 + loss3
            # loss = -torch.mean(sim(lab_hat, new_labels[lab]))
            # loss = criterion(lab_hat, lab, new_labels)
            
            loss.backward()
            optimizer1.step()
            optimizer2.step()
            if xi % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss1: {:.6f},\tLoss2: {:.6f},\tLoss3: {:.6f}'.format(
                    epoch, xi * len(dat), len(train_loader.dataset),
                    100. * xi / len(train_loader), loss1.item(), loss2.item(), loss3.item()))

        with torch.no_grad():
            correct = 0
            for xi, x in enumerate(test_loader):
                dat, lab = x
                dat = dat.to(device)
                lab = lab.to(device)
                feature_batch = model(dat)
                
                batch_size, feature_size = feature_batch.shape
                tiled_f = feature_batch.repeat(1,10).view(batch_size, -1, feature_size)
                vectors = l2v(torch.LongTensor([0,1,2,3,4,5,6,7,8,9]).to(device))
                class_num, feature_size = vectors.shape
                tiled_v = vectors.repeat(batch_size, 1).view(batch_size, class_num, feature_size)
                tiled_sim_f = sim_(tiled_f, tiled_v) # 256 * 10
                chu = torch.argmax(tiled_sim_f, dim=1)
                correct += torch.sum(torch.eq(chu, lab))
            print("Test accuracy:{:.1f}".format(correct/len(test_loader.dataset)*100))
            
            if (epoch+1) % 10 ==0:
                correct = 0
                for xi, x in enumerate(train_loader):
                    dat, lab = x
                    dat = dat.to(device)
                    lab = lab.to(device)
                    feature_batch = model(dat)
                    
                    batch_size, feature_size = feature_batch.shape
                    tiled_f = feature_batch.repeat(1,10).view(batch_size, -1, feature_size)
                    vectors = l2v(torch.LongTensor([0,1,2,3,4,5,6,7,8,9]).to(device))
                    class_num, feature_size = vectors.shape
                    tiled_v = vectors.repeat(batch_size, 1).view(batch_size, class_num, feature_size)
                    tiled_sim_f = sim_(tiled_f, tiled_v) # 256 * 10
                    chu = torch.argmax(tiled_sim_f, dim=1)
                    correct += torch.sum(torch.eq(chu, lab))
                print("Train accuracy:{:.1f}".format(correct/len(train_loader.dataset)*100))
        
                
                

