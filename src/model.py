import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from src.utils import get_file, savencommit
from torchvision import models
from byol_pytorch import BYOL
from pl_bolts.optimizers.lars_scheduling import LARSWrapper
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from scheduler import WarmupCosineLR
from pytorch_lightning.metrics import Accuracy

savencommit(__file__)

class BYOL_Pre(pl.LightningModule):
    #################################################################################
    # The model used for pretraining an encoder using BYOL.
    #################################################################################
    def __init__(self, hparams, *args, **kwargs):
        super().__init__()
        self.hparams = hparams
        self.model = models.resnet18(pretrained=self.hparams.pretrain)
        self.learner = BYOL(
            self.model,
            image_size = hparams.image_size,
            hidden_layer = 'avgpool'
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_index):
        x, y = batch
        loss = self.learner(x)
        self.log('train_loss', loss, on_epoch=True)
        
        return loss

    def on_train_batch_end(self, outputs, batch, batch_index, dataloader):
        self.learner.update_moving_average()

    def validation_step(self, batch, batch_index):
        x, y = batch
        loss = self.learner(x)
        self.log('val_loss', loss, on_epoch=True)
        
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        optimizer = LARSWrapper(optimizer)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=self.hparams.warmup_epochs, max_epochs=self.hparams.max_epochs
        )
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # optim
        parser.add_argument('--learning_rate', type=float, default=3e-4)
        parser.add_argument('--weight_decay', type=float, default=1.5e-6)
        parser.add_argument('--warmup_epochs', type=float, default=10)
        parser.add_argument('--pretrain', type=bool, default=False)
        parser.add_argument('--image_size', type=int, default=96)
        return parser

class Classifier(pl.LightningModule):
    #################################################################################
    # The main classifier. 
    #################################################################################
    def __init__(self, hparams, model, *args, **kwargs):
        super().__init__()
        self.hparams = hparams
        
        self.model = model
        self.accuracy = Accuracy()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_index):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        accuracy = self.accuracy(y_hat, y)
        self.log('loss/train', loss)
        self.log('acc/train', accuracy)
        return loss

    def validation_step(self, batch, batch_index):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        accuracy = self.accuracy(y_hat, y)
        self.log('loss/val', loss)
        self.log('acc/val', accuracy)


    def test_step(self, batch, batch_index):
        pass

    def configure_optimizers(self):
        if self.hparams.optimizer == 'adam':
            optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=self.hparams.learning_rate, 
                weight_decay=self.hparams.weight_decay
            )
            return optimizer
        elif self.hparams.optimizer == 'sgd':
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
                momentum=0.9,
                nesterov=True,
            )
        total_steps = self.hparams.max_epochs * len(self.train_dataloader())
        scheduler = {
            "scheduler": WarmupCosineLR(
                optimizer, warmup_epochs=total_steps * 0.3, max_epochs=total_steps
            ),
            "interval": "step",
            "name": "learning_rate",
        }
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--optimizer", type=str, default='adam')
        parser.add_argument("--learning_rate", type=float, default=1e-4)
        parser.add_argument("--weight_decay", type=float, default=1e-6)
        
        return parser
        

class Classifier_new(pl.LightningModule):
    #################################################################################
    # The main classifier. 
    #################################################################################
    def __init__(self, model, labels, *args, **kwargs):
        super().__init__()
        self.criterion = nn.CosineSimilarity()
        self.model = model
        self.labels = labels
        self.accuracy = Accuracy()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_index):
        x, y = batch
        y_hat = self.forward(x)
        loss = -torch.mean(self.criterion(y_hat, self.labels[y]))
        with torch.no_grad():
            y_hats = y_hat.repeat(10,1)
            accuracy = self.accuracy(torch.argmax(y_hats), y)
        self.log('loss/train', loss)
        self.log('acc/train', accuracy)
        return loss

    def validation_step(self, batch, batch_index):
        x, y = batch
        y_hat = self.forward(x)
        loss = -torch.mean(self.criterion(y_hat, self.labels[y]))
        with torch.no_grad():
            print(self.criterion(y_hats, self.labels).shape)
            accuracy = self.accuracy(torch.argmax(self.criterion(y_hats, self.labels)), y)
        self.log('loss/train', loss)
        self.log('acc/train', accuracy)
        return loss


    def test_step(self, batch, batch_index):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(
                self.model.parameters(), 
                lr=1e-3
            )
        # if self.hparams.optimizer == 'adam':
        #     optimizer = torch.optim.Adam(
        #         self.model.parameters(), 
        #         lr=self.hparams.learning_rate, 
        #         weight_decay=self.hparams.weight_decay
        #     )
        #     return optimizer
        # elif self.hparams.optimizer == 'sgd':
        #     optimizer = torch.optim.SGD(
        #         self.model.parameters(),
        #         lr=self.hparams.learning_rate,
        #         weight_decay=self.hparams.weight_decay,
        #         momentum=0.9,
        #         nesterov=True,
        #     )
        # total_steps = self.hparams.max_epochs * len(self.train_dataloader())
        # scheduler = {
        #     "scheduler": WarmupCosineLR(
        #         optimizer, warmup_epochs=total_steps * 0.3, max_epochs=total_steps
        #     ),
        #     "interval": "step",
        #     "name": "learning_rate",
        # }
        # return [optimizer], [scheduler]

    # @staticmethod
    # def add_model_specific_args(parent_parser):
    #     parser = ArgumentParser(parents=[parent_parser], add_help=False)
    #     parser.add_argument("--optimizer", type=str, default='adam')
    #     parser.add_argument("--learning_rate", type=float, default=1e-4)
    #     parser.add_argument("--weight_decay", type=float, default=1e-6)
        
    #     return parser
        
