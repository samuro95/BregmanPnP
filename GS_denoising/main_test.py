import pytorch_lightning as pl
from lightning_denoiser import GradMatch
from data_module import DataModule
from pytorch_lightning import loggers as pl_loggers
import os
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from argparse import ArgumentParser
import random
import torch

if __name__ == '__main__':

    # PROGRAM args
    parser = ArgumentParser()
    parser.add_argument('--name', type=str, default='my_test')
    parser.add_argument('--save_images', dest='save_images', action='store_true')
    parser.set_defaults(save_images=False)
    parser.add_argument('--log_folder', type=str, default='logs')
    

    # MODEL args
    parser = GradMatch.add_model_specific_args(parser)
    # DATA args
    parser = DataModule.add_data_specific_args(parser)
    # OPTIM args
    parser = GradMatch.add_optim_specific_args(parser)

    hparams = parser.parse_args()

    random.seed(0)

    if not os.path.exists(hparams.log_folder):
        os.mkdir(hparams.log_folder)
    log_path = hparams.log_folder + '/' + hparams.name
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    tb_logger = pl_loggers.TensorBoardLogger(log_path)

    model = GradMatch(hparams)
    dm = DataModule(hparams)

    model.test_dataloader = dm.val_dataloader

    checkpoint = torch.load(hparams.pretrained_checkpoint)
    model.load_state_dict(checkpoint['state_dict'],strict=False)

    trainer = pl.Trainer(logger=tb_logger,gpus=-1,precision=32, accelerator="gpu", strategy="ddp")

    trainer.test(model)




