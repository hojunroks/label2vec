import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import get_transformer_encoder, get_resnet18_encoder
from argparse import ArgumentParser


class AttentionClassifier(pl.LightningModule):
    #################################################################################
    # The main classifier. First goes through a series of residual blocks for\
    # dimension increase, and then transformer encoder, and then a linear layer.
    #################################################################################
    def __init__(self, hparams, *args, **kwargs):
        super().__init__()
        self.hparams = hparams
        # self.resnet18_encoder = get_resnet18_encoder()
        self.trans_encoder = get_transformer_encoder(d_model=hparams.encoder_dim, nhead=hparams.nhead, num_layers=hparams.encoder_layers)
        self.fc = nn.Sequential(
            nn.Linear()
        )

    def forward(self, x):
        # x = self.resnet18_encoder(x)
        b, c, h, w = x.shape
        x = x.reshape(b, c, h*w)   # b*c*hw
        x = x.permute(2, 0, 1)     # hw*b*c
        x = self.trans_encoder(x)
        print(x.shape)
        return x

    def training_step(self, batch, batch_index):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_index):
        pass

    def test_step(self, batch, batch_index):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--encoder_dim', default=3, type=int)
        parser.add_argument('--nhead', default=3, type=int)
        parser.add_argument('--encoder_layers', default=12, type=int)
        parser.add_argument('--learning_rate', default=1e-3, type=float)
        return parser