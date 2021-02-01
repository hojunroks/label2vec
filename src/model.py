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

# class Classifier(pl.LightningModule):
#     #################################################################################
#     # The main classifier. 
#     #################################################################################
#     def __init__(self, hparams, model, *args, **kwargs):
#         super().__init__()
#         self.hparams = hparams
        
#         self.model = model
#         self.accuracy = Accuracy()

#     def forward(self, x):
#         return self.model(x)

#     def training_step(self, batch, batch_index):
#         x, y = batch
#         y_hat = self.forward(x)
#         loss = F.cross_entropy(y_hat, y)
#         accuracy = self.accuracy(y_hat, y)
#         self.log('loss/train', loss)
#         self.log('acc/train', accuracy)
#         return loss

#     def validation_step(self, batch, batch_index):
#         x, y = batch
#         y_hat = self.forward(x)
#         loss = F.cross_entropy(y_hat, y)
#         accuracy = self.accuracy(y_hat, y)
#         self.log('loss/val', loss)
#         self.log('acc/val', accuracy)


#     def test_step(self, batch, batch_index):
#         pass

#     def configure_optimizers(self):
#         if self.hparams.optimizer == 'adam':
#             optimizer = torch.optim.Adam(
#                 self.model.parameters(), 
#                 lr=self.hparams.learning_rate, 
#                 weight_decay=self.hparams.weight_decay
#             )
#         elif self.hparams.optimizer == 'sgd':
#             optimizer = torch.optim.SGD(
#                 self.model.parameters(),
#                 lr=self.hparams.learning_rate,
#                 weight_decay=self.hparams.weight_decay,
#                 momentum=0.9,
#                 nesterov=True,
#             )
#         total_steps = self.hparams.max_epochs * len(self.train_dataloader())
#         scheduler = {
#             "scheduler": WarmupCosineLR(
#                 optimizer, warmup_epochs=total_steps * 0.3, max_epochs=total_steps
#             ),
#             "interval": "step",
#             "name": "learning_rate",
#         }
#         return [optimizer], [scheduler]

#     @staticmethod
#     def add_model_specific_args(parent_parser):
#         parser = ArgumentParser(parents=[parent_parser], add_help=False)
#         parser.add_argument("--optimizer", type=str, default='adam')
#         parser.add_argument("--learning_rate", type=float, default=1e-4)
#         parser.add_argument("--weight_decay", type=float, default=1e-6)
        
#         return parser



class Classifier(pl.LightningModule):
    #################################################################################
    # The main classifier. 
    #################################################################################
    def __init__(self, hparams, *args, **kwargs):
        super().__init__()
        self.hparams = hparams

        self.model = models.resnet18(pretrained=hparams.pretrained_resnet)
        pre_file = get_file(hparams.pretrained_code + '.ckpt')
        if pre_file is not None:
            byol = BYOL_Pre.load_from_checkpoint(pre_file)
            self.model.load_state_dict(byol.fe.state_dict())
        
        # self.model = kwargs['feature_extractor']
        # self.model = models.resnet18(pretrained=False)
        # for name, param in self.model.named_parameters():
        #     if name not in ['fc.weight', 'fc.bias']:
        #         param.requires_grad = False

        # self.model.fc = nn.Linear(self.model.fc.in_features, 10)
        # self.fine_parameters = list(filter(lambda p: p.requires_grad, self.model.parameters()))
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
        # correct=y_hat.argmax(dim=1).eq(y).sum().item()
        # total = len(y)
        self.log('loss/val', loss)
        self.log('acc/val', accuracy)

        # batch_dictionary={
        #     "orig_img": x,
        #     "prediction": y_hat,
        #     "truth": y,
        #     "loss": loss,
        #     "correct": correct,
        #     "total": total
        # }
        # return batch_dictionary
        
    # def validation_epoch_end(self, val_step_outputs):
    #     # do something with all the predictions from each validation_step
    #     avg_loss = torch.stack([x['loss'] for x in val_step_outputs]).mean()
    #     correct = sum([x["correct"] for x in val_step_outputs])
    #     total = sum([x["total"] for x in val_step_outputs])
    #     self.logger.experiment.add_scalar("Loss/validation", avg_loss, self.current_epoch)
    #     self.logger.experiment.add_scalar("Loss/accuracy", correct/total, self.current_epoch)

    #     orig_images = torch.stack([output["orig_img"][0] for output in val_step_outputs])
    #     labels = [output["prediction"][0] for output in val_step_outputs]
    #     idx = random.sample(range(len(orig_images)), min(len(orig_images), 8))
    #     fig = plt.figure(figsize=(9, 9))
    #     if len(idx)==4:
    #         for i in range(4):
    #             color = "red"
    #             pred = np.argmax(np.array(labels[idx[i]].cpu()))
    #             truth = "(" + self.class_list[val_step_outputs[idx[i]]['truth'][0]] + ")"
    #             if pred==val_step_outputs[idx[i]]['truth'][0]:
    #                 color = "blue"
    #                 truth = ""
    #             ax = fig.add_subplot(2,2,i+1)
    #             if self.input_channels == 3 or self.input_channels == 4:
    #                 plt.imshow((orig_images[idx[i]].permute(1, 2, 0).cpu()*255).type(torch.int))
    #             elif self.input_channels == 1:
    #                 plt.imshow(orig_images[idx[i]][0].cpu())
    #             ax.set_title("{}{}".format(self.class_list[pred], truth), color=color)
    #     plt.tight_layout()

    #     self.logger.experiment.add_figure('figure', fig, global_step=self.current_epoch)

    #     return

    def test_step(self, batch, batch_index):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-6)
        return optimizer
        # optimizer = torch.optim.SGD(
        #     self.model.parameters(),
        #     lr=self.hparams.learning_rate,
        #     weight_decay=self.hparams.weight_decay,
        #     momentum=0.9,
        #     nesterov=True,
        # )
        # total_steps = self.hparams.max_epochs * len(self.train_dataloader())
        # scheduler = {
        #     "scheduler": WarmupCosineLR(
        #         optimizer, warmup_epochs=total_steps * 0.3, max_epochs=total_steps
        #     ),
        #     "interval": "step",
        #     "name": "learning_rate",
        # }
        # return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", type=float, default=1e-2)
        parser.add_argument("--weight_decay", type=float, default=1e-2)
        parser.add_argument("--optimizer", type=str, default='adam')        
        return parser


