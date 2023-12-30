from module import Module
from cnn import Net as CNN
from labels import prepare_labels, prepare_labels_short
from dataset import Dataset, train_files, train_dimension

import pytorch_lightning as pl
import os
import torch
import numpy as np


def training(case):
    if case == 1000:
        train_path = "G:/data/train_data_npy_100/1727"
        train_path_folder = "G:/data/train_data_npy_100"
    if case == 100:
        train_path = "G:/data/train_data_npy_10/1727"
        train_path_folder = "G:/data/train_data_npy_10"
    if case == 10:
        train_path = "G:/data/train_data_npy/1727"
        train_path_folder = "G:/data/train_data_npy"

    # START [TRAINING]

    train_files_path = os.listdir(train_path)

    # train files
    for file in range(len(train_files_path)):
        file_path = os.path.join(train_path, train_files_path[file])
        melspectrogram = np.load(file_path)
        train_files.append(melspectrogram)

    train_files_labels_family = prepare_labels(0, case, train_path_folder)[0]
    train_files_labels_instruments = prepare_labels(0, case, train_path_folder)[1]
    train_dataset = Dataset(
        train_files,
        labels_family=train_files_labels_family,
        labels_instruments=train_files_labels_instruments,
        dimension=train_dimension,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True
    )

    # END [TRAINING] ----------------------

    true_labels_instruments = prepare_labels_short(1)
    true_labels_family = prepare_labels_short(0)

    learner = Module(
        model_family=CNN(4, case),
        model_instruments=CNN(11, case),
        true_labels_instruments=true_labels_instruments,
        true_labels_family=true_labels_family,
    )

    # logger = TensorBoardLogger("logs/", name="logger")
    checkpoint = pl.callbacks.ModelCheckpoint(monitor="loss")
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=100,
        callbacks=[checkpoint],
        # logger=logger,
        log_every_n_steps=50,
    )
    trainer.fit(learner, train_dataloaders=train_loader)
    # trainer.test(model=learner, dataloaders=test_loader)


# [ 1000 = 1 [sec] | 100 = 100 [ms] | 10 = 10 [ms] ]
# training(1000)
# training(100)
training(10)
