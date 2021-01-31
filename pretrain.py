from argparse import ArgumentParser
from src.model import Classifier, BYOL_Pre
import pytorch_lightning as pl
from pl_bolts.datamodules import CIFAR10DataModule, STL10DataModule
from pl_bolts.models.self_supervised import BYOL
from pl_bolts.models.self_supervised.simclr import SimCLREvalDataTransform, SimCLRTrainDataTransform
import torch
from datetime import datetime
from torchvision import models
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import random
import string
from src.utils import savencommit

def main():
    print("START PROGRAM")
    
    #######################
    # GIT
    #######################
    print("CONFIGURING GIT...")
    repo = savencommit(__file__)


    #######################
    # PARSE ARGUMENTS
    #######################
    print("PARSING ARGUMENTS...")
    parser = ArgumentParser()
    
    # add PROGRAM level args
    parser.add_argument('--dataset', default='stl10', type=str)
    parser.add_argument('--commit', default=repo.head.commit, type=str)

    # add all the available trainer options to argparse
    # ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
    parser = pl.Trainer.add_argparse_args(parser)

    # add model specific args
    parser = BYOL_Pre.add_model_specific_args(parser)

    args = parser.parse_args()

    ###########################
    # INITIALIZE DATAMODULE
    ###########################
    print("INITIALIZING DATAMODULE...")
    if args.dataset=='stl10':
        dm = STL10DataModule(data_dir='./data', batch_size=128)
    elif args.dataset=='cifar10':
        dm = CIFAR10DataModule(data_dir='./data')
    
    ###########################r
    # INITIALIZE MODEL
    ###########################
    print("INITIALIZING MODEL...")
    model = BYOL_Pre(args)


    ###########################
    # INITIALIZE LOGGER
    ###########################
    print("INITIALIZING LOGGER...")
    logdir = 'logs'
    logdir += datetime.now().strftime("/%m%d")
    logdir += '/byol'
    logdir += '/{}'.format(args.dataset)
    if args.pretrain:
        logdir += '/pretrained'
    logdir += '/{}epochs'.format(args.max_epochs)
    code = ""
    for i in range(4):
        code += string.ascii_uppercase[random.randint(0,25)]
    logger = TensorBoardLogger(logdir, name='', version=code)


    ###########################
    # TRAIN
    ###########################
    print("START TRAINING...")
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=logger.log_dir+logger.name+"/",
        filename='{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min',
    )

    trainer = pl.Trainer.from_argparse_args(args, logger=logger, callbacks=[checkpoint_callback])
    trainer.fit(model, datamodule=dm)


    trainer.save_checkpoint(logger.log_dir+logger.name+"/"+code+".ckpt")
    
    ###########################
    # TEST
    ###########################
    print("START TESTING...")
    # result = trainer.test(datamodule=dm)
    # print(result)

if __name__=='__main__':
    main()