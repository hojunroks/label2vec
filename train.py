from argparse import ArgumentParser
from src.model import Classifier, BYOL_Pre, CIFAR10Module
import pytorch_lightning as pl
from pl_bolts.datamodules import CIFAR10DataModule, STL10DataModule
from pl_bolts.models.self_supervised import BYOL
from pl_bolts.models.self_supervised.simclr import SimCLREvalDataTransform, SimCLRTrainDataTransform
import torch
from torchvision import models, datasets
from datetime import datetime
from src.utils import get_file, savencommit
from pytorch_lightning.loggers import TensorBoardLogger
import git
from dm import CIFAR10Data

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
    parser.add_argument('--pretrained_code', default='', type=str)
    parser.add_argument('--dataset', default='stl10', type=str)
    parser.add_argument('--pretrained_resnet', default=False, type=bool)
    parser.add_argument('--commit', default=repo.head.commit, type=str)

    # add all the available trainer options to argparse
    # ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
    parser = pl.Trainer.add_argparse_args(parser)

    # add model specific args
    parser = Classifier.add_model_specific_args(parser)

    args = parser.parse_args()

    ###########################
    # INITIALIZE DATAMODULE
    ###########################
    print("INITIALIZING DATAMODULE...")

    if args.dataset=='stl10':
        dm = STL10DataModule(data_dir='./data', batch_size=128)
        dm.train_dataloader = dm.train_dataloader_labeled
        dm.val_dataloader = dm.val_dataloader_labeled
    elif args.dataset=='cifar10':
        dm = CIFAR10DataModule(data_dir='./data', batch_size=256, num_workers=8)
    
    ###########################r
    # LOAD PRETRAINED MODEL
    ###########################
    print("LOADING PRETRAINED MODEL...")
    pre_file = get_file(args.pretrained_code + '.ckpt')
    
    # fe = models.resnet18(pretrained=False)
    
    ###########################
    # INITIALIZE MODEL
    ###########################
    print("INITIALIZING MODEL...")
    model = models.resnet18(pretrained=args.pretrained_resnet)
    if pre_file is not None:
        args.image_size=32
        byol = BYOL_Pre.load_from_checkpoint(pre_file)
        model.load_state_dict(byol.fe.state_dict())
    classifier = Classifier(args, model=model)

    ###########################
    # INITIALIZE LOGGER
    ###########################
    print("INITIALIZING LOGGER...")
    logdir = 'logs'
    logdir += datetime.now().strftime("/%m%d")
    logdir += '/finetuned'
    logdir += '/{}'.format(args.dataset)
    logdir += '/{}epochs'.format(args.max_epochs)
    logdir += '/{}'.format(args.optimizer)
    if pre_file is not None:
        logger = TensorBoardLogger(logdir, name=args.pretrained_code)
    else:
        logdir += '/no_byol'
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
    dm = CIFAR10Data(args)
    classifier = CIFAR10Module(args)
    trainer.fit(classifier, datamodule=dm
    )
    if pre_file is not None:
        trainer.save_checkpoint(logger.log_dir+args.pretrained_code+"_finetuned.ckpt")
    else:
        trainer.save_checkpoint(logger.log_dir+logger.name+"/"+logger.name+"resnet_finetuned.ckpt")

    ###########################
    # TEST
    ###########################
    print("START TESTING...")
    # result = trainer.test(datamodule=dm)
    # print(result)

if __name__=='__main__':
    main()