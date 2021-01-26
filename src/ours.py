import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from torchvision import models
from byol_pytorch import BYOL
from pl_bolts.optimizers.lars_scheduling import LARSWrapper
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

class Ours(pl.LightningModule):
    #################################################################################
    # The main classifier using different target coding.
    #################################################################################
    def __init__(self, hparams, *args, **kwargs):
        super().__init__()
        self.hparams = hparams
        self.model = kwargs['feature_extractor']
        self.labels_mean = kwargs['target_code_mean']
        self.labels_std = kwargs['target_code_std']
        # for name, param in self.model.named_parameters():
        #     if name not in ['fc.weight', 'fc.bias']:
        #         param.requires_grad = False

        # self.model.fc = nn.Linear(self.model.fc.in_features, 10)
        # self.fine_parameters = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        self.cos = nn.CosineSimilarity()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_index):
        x, y = batch
        y_hat = self.forward(x)
        
        loss = nn.CosineSimilarity(y_hat, self.labels_avg[y])/self.labels_std[y]
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_index):
        x, y = batch
        y_hat = self.forward(x)
        loss = nn.CosineSimilarity(y_hat, self.labels_avg[y])/self.labels_std[y]
        
        correct=y_hat.argmax(dim=1).eq(y).sum().item()
        total = len(y)


        batch_dictionary={
            "orig_img": x,
            "prediction": y_hat,
            "truth": y,
            "loss": loss,
            "correct": correct,
            "total": total
        }
        return batch_dictionary
        
    def validation_epoch_end(self, val_step_outputs):
        # do something with all the predictions from each validation_step
        avg_loss = torch.stack([x['loss'] for x in val_step_outputs]).mean()
        correct = sum([x["correct"] for x in val_step_outputs])
        total = sum([x["total"] for x in val_step_outputs])
        self.logger.experiment.add_scalar("Loss/validation", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar("Loss/accuracy", correct/total, self.current_epoch)

        orig_images = torch.stack([output["orig_img"][0] for output in val_step_outputs])
        labels = [output["prediction"][0] for output in val_step_outputs]
        idx = random.sample(range(len(orig_images)), min(len(orig_images), 8))
        fig = plt.figure(figsize=(9, 9))
        if len(idx)==4:
            for i in range(4):
                color = "red"
                pred = np.argmax(np.array(labels[idx[i]].cpu()))
                truth = "(" + self.class_list[val_step_outputs[idx[i]]['truth'][0]] + ")"
                if pred==val_step_outputs[idx[i]]['truth'][0]:
                    color = "blue"
                    truth = ""
                ax = fig.add_subplot(2,2,i+1)
                if self.input_channels == 3 or self.input_channels == 4:
                    plt.imshow((orig_images[idx[i]].permute(1, 2, 0).cpu()*255).type(torch.int))
                elif self.input_channels == 1:
                    plt.imshow(orig_images[idx[i]][0].cpu())
                ax.set_title("{}{}".format(self.class_list[pred], truth), color=color)
        plt.tight_layout()

        self.logger.experiment.add_figure('figure', fig, global_step=self.current_epoch)

        return

    def test_step(self, batch, batch_index):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-6)
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', default=1e-3, type=float)
        
        return parser