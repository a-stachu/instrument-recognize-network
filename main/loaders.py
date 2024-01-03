import os
import torch
import numpy as np
from dataset import (
    Dataset,
    train_dimension,
    test_files,
    train_files,
    test_files_mfcc,
    train_files_mfcc,
)
from labels import prepare_labels


def load_training_data(case):
    if case == 1000:
        train_path = "G:/data/train_data_npy_100/1727"
        train_path_folder = "G:/data/train_data_npy_100"
        train_path_mfcc = "G:/data/train_data_npy_100_mfcc/1727"

    if case == 100:
        train_path = "G:/data/train_data_npy_10/1727"
        train_path_folder = "G:/data/train_data_npy_10"
        train_path_mfcc = "G:/data/train_data_npy_10_mfcc/1727"

    if case == 10:
        train_path = "G:/data/train_data_npy/1727"
        train_path_folder = "G:/data/train_data_npy"
        train_path_mfcc = "G:/data/train_data_npy_mfcc/1727"

    # START [TRAINING]

    train_files_path = os.listdir(train_path)
    train_files_path_mfcc = os.listdir(train_path_mfcc)

    # train files
    for file in range(len(train_files_path)):
        file_path = os.path.join(train_path, train_files_path[file])
        file_path_mfcc = os.path.join(train_path_mfcc, train_files_path_mfcc[file])
        melspectrogram = np.load(file_path)
        mfcc = np.load(file_path_mfcc)
        train_files.append(melspectrogram)
        train_files_mfcc.append(mfcc)

    train_files_labels_family = prepare_labels(0, case, train_path_folder, "train")[0]
    train_files_labels_instruments = prepare_labels(
        0, case, train_path_folder, "train"
    )[1]
    train_dataset = Dataset(
        train_files,
        train_files_mfcc,
        labels_family=train_files_labels_family,
        labels_instruments=train_files_labels_instruments,
        dimension=train_dimension,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True
    )
    return train_loader

    # END [TRAINING] ----------------------


def load_test_data(case):
    if case == 1000:
        test_path = "G:/data/test_data_npy_100/1759"
        test_path_folder = "G:/data/test_data_npy_100"
        test_path_mfcc = "G:/data/test_data_npy_100_mfcc/1759"
    if case == 100:
        test_path = "G:/data/test_data_npy_10/1759"
        test_path_folder = "G:/data/test_data_npy_10"

        test_path_mfcc = "G:/data/test_data_npy_10_mfcc/1759"
    if case == 10:
        test_path = "G:/data/test_data_npy/1759"
        test_path_folder = "G:/data/test_data_npy"

        test_path_mfcc = "G:/data/test_data_npy_mfcc/1759"

    # START [TEST]

    test_files_path_mel = os.listdir(test_path)
    test_files_path_mfcc = os.listdir(test_path_mfcc)

    # test files
    for file in range(len(test_files_path_mel)):
        file_path_mel = os.path.join(test_path, test_files_path_mel[file])
        file_path_mfcc = os.path.join(test_path_mfcc, test_files_path_mfcc[file])
        melspectrogram = np.load(file_path_mel)
        mfcc = np.load(file_path_mfcc)
        test_files.append(melspectrogram)
        test_files_mfcc.append(mfcc)

    test_files_labels_family = prepare_labels(0, case, test_path_folder, "test")[0]
    test_files_labels_instruments = prepare_labels(0, case, test_path_folder, "test")[1]
    test_dataset = Dataset(
        test_files,
        test_files_mfcc,
        labels_family=test_files_labels_family,
        labels_instruments=test_files_labels_instruments,
        dimension=train_dimension,
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)
    return test_loader

    # END [TEST] ----------------------
