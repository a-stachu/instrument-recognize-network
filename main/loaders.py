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
from helpers import list_files_in_subdirectories


def load_training_data(case):
    if case == 1000:
        train_path = "G:/data/train_data_npy_100"
        train_path_folder = "G:/data/train_data_npy_100"
        train_path_mfcc = "G:/data/train_data_npy_100_mfcc"

    if case == 100:
        train_path = "G:/data/train_data_npy_10"
        train_path_folder = "G:/data/train_data_npy_10"
        train_path_mfcc = "G:/data/train_data_npy_10_mfcc"

    if case == 10:
        train_path = "G:/data/train_data_npy"
        train_path_folder = "G:/data/train_data_npy"
        train_path_mfcc = "G:/data/train_data_npy_mfcc"

    # START [TRAINING]

    train_files_path = list_files_in_subdirectories(train_path)
    train_files_path_mfcc = list_files_in_subdirectories(train_path_mfcc)

    # train files
    for segment in range(len(train_files_path)):
        file_path = os.path.join(train_path, train_files_path[segment])
        file_path_mfcc = os.path.join(train_path_mfcc, train_files_path_mfcc[segment])
        # print(file_path)
        melspectrogram = np.load(file_path)
        mfcc = np.load(file_path_mfcc)
        train_files.append(melspectrogram)
        train_files_mfcc.append(mfcc)

    train_files_labels_family = []
    train_files_labels_instruments = []

    for file in range(len(os.listdir(train_path))):
        train_files_labels_family.append(
            prepare_labels(file, case, train_path_folder, "train")[0]
        )
        train_files_labels_instruments.append(
            prepare_labels(file, case, train_path_folder, "train")[1]
        )  # family 0 inst 1

    train_files_labels_family = [
        label for row in train_files_labels_family for label in row
    ]
    train_files_labels_instruments = [
        label for row in train_files_labels_instruments for label in row
    ]
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
        test_path = "G:/data/test_data_npy_100"
        test_path_folder = "G:/data/test_data_npy_100"
        test_path_mfcc = "G:/data/test_data_npy_100_mfcc"
    if case == 100:
        test_path = "G:/data/test_data_npy_10"
        test_path_folder = "G:/data/test_data_npy_10"
        test_path_mfcc = "G:/data/test_data_npy_10_mfcc"
    if case == 10:
        test_path = "G:/data/test_data_npy"
        test_path_folder = "G:/data/test_data_npy"
        test_path_mfcc = "G:/data/test_data_npy_mfcc"

    # START [TEST]

    test_files_path = list_files_in_subdirectories(test_path)
    test_files_path_mfcc = list_files_in_subdirectories(test_path_mfcc)

    # test files
    for file in range(len(test_files_path)):
        file_path = os.path.join(test_path, test_files_path[file])
        # print(file_path)
        file_path_mfcc = os.path.join(test_path_mfcc, test_files_path_mfcc[file])
        melspectrogram = np.load(file_path)
        mfcc = np.load(file_path_mfcc)
        test_files.append(melspectrogram)
        test_files_mfcc.append(mfcc)

    test_files_labels_family = []
    test_files_labels_instruments = []

    for file in range(len(os.listdir(test_path))):
        test_files_labels_family.append(
            prepare_labels(file, case, test_path_folder, "test")[0]
        )
        test_files_labels_instruments.append(
            prepare_labels(file, case, test_path_folder, "test")[1]
        )  # family 0 inst 1

    test_files_labels_family = [
        label for row in test_files_labels_family for label in row
    ]
    test_files_labels_instruments = [
        label for row in test_files_labels_instruments for label in row
    ]

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
