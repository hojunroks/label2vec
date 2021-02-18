from argparse import ArgumentParser
from src.model import Classifier, BYOL_Pre, CIFAR10Module, Ours, Identity
import pytorch_lightning as pl
from pl_bolts.datamodules import CIFAR10DataModule, STL10DataModule
from pl_bolts.models.self_supervised import BYOL
from pl_bolts.models.self_supervised.simclr import SimCLREvalDataTransform, SimCLRTrainDataTransform
import torch
from torchvision import models, datasets
from datetime import datetime
from src.utils import get_file, savencommit
from src.resnet import resnet18
from pytorch_lightning.loggers import TensorBoardLogger
import git
from dm import CIFAR10Data

def main():
    print("START PROGRAM")

    #######################
    # GIT
    #######################
    # print("CONFIGURING GIT...")
    # repo = savencommit(__file__)

    #######################
    # PARSE ARGUMENTS
    #######################
    print("PARSING ARGUMENTS...")
    parser = ArgumentParser()
    
    # add PROGRAM level args
    # parser.add_argument('--commit', default=repo.head.commit, type=str)

    # add all the available trainer options to argparse
    # ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
    parser = pl.Trainer.add_argparse_args(parser)

    # add model specific args
    parser = Ours.add_model_specific_args(parser)

    args = parser.parse_args()

    ###########################
    # INITIALIZE DATAMODULE
    ###########################
    # print("INITIALIZING DATAMODULE...")

    # if args.dataset=='stl10':
    #     dm = STL10DataModule(data_dir='./data', batch_size=128)
    #     dm.train_dataloader = dm.train_dataloader_labeled
    #     dm.val_dataloader = dm.val_dataloader_labeled
    # elif args.dataset=='cifar10':
    #     dm = CIFAR10DataModule(data_dir='./data', batch_size=512, num_workers=8)
    
    ###########################r
    # LOAD PRETRAINED MODEL
    ###########################
    # print("LOADING PRETRAINED MODEL...")
    # pre_file = get_file(args.pretrained_code + '.ckpt')
    
    # fe = models.resnet18(pretrained=False)
    
    ###########################
    # INITIALIZE MODEL
    ###########################
    print("INITIALIZING MODEL...")
    # model = resnet18()['backbone']
    model = resnet18()
    model.fc = Identity()
    classifier = Ours(args, model=model)

    ###########################
    # INITIALIZE LOGGER
    ###########################
    print("INITIALIZING LOGGER...")
    logdir = 'logs_ours'
    logdir += datetime.now().strftime("/%m%d")
    logdir += '/ours'
    # logdir += '/{}'.format(args.dataset)
    logdir += '/{}epochs'.format(args.max_epochs)
    logger = TensorBoardLogger(logdir, name='')


    ###########################
    # TRAIN
    ###########################
    print("START TRAINING...")
    trainer = pl.Trainer.from_argparse_args(args,
        logger=logger,
        fast_dev_run=False,
        deterministic=True,
        weights_summary=None,
        log_every_n_steps=1
    )
    args.data_dir = './data'
    args.batch_size = 512
    args.num_workers = 8
    dm = CIFAR10Data(args)
    # classifier = CIFAR10Module(args)
    trainer.fit(classifier, datamodule=dm)
    trainer.save_checkpoint(logger.log_dir+"_done.ckpt")
    

    ###########################
    # TEST
    ###########################
    print("START TESTING...")
    result = trainer.test(datamodule=dm)
    print(result)

if __name__=='__main__':
    main()