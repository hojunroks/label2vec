from argparse import ArgumentParser
from src.model import AttentionClassifier
import pytorch_lightning as pl
from pl_bolts.datamodules import CIFAR10DataModule

def main():
    print("START PROGRAM")
    #######################
    # PARSE ARGUMENTS
    #######################
    print("PARSING ARGUMENTS...")
    parser = ArgumentParser()
    # add PROGRAM level args
    # parser.add_argument('--whatever', default=128, type=int)

    # add all the available trainer options to argparse
    # ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
    parser = pl.Trainer.add_argparse_args(parser)

    # add model specific args
    parser = AttentionClassifier.add_model_specific_args(parser)

    args = parser.parse_args()

    ###########################
    # INITIALIZE DATAMODULE
    ###########################
    print("INITIALIZING DATAMODULE...")
    datamodule = CIFAR10DataModule(data_dir='./data')

    ###########################
    # INITIALIZE MODEL
    ###########################
    print("INITIALIZING MODEL...")
    model = AttentionClassifier(args)

    ###########################
    # TRAIN
    ###########################
    print("START TRAINING...")
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, datamodule=datamodule)

    ###########################
    # TEST
    ###########################
    print("START TESTING...")
    result = trainer.test(datamodule=datamodule)
    print(result)

if __name__=='__main__':
    main()