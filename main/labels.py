import pandas as pd
import os
import numpy as np
import math

from instruments import *

from helpers import *


# one label for one segment of melspec
def prepare_labels(index):
    num_instruments_family = len(instruments_map_arr)
    num_instruments = len(instruments)
    empty_one_hot_family = np.zeros(num_instruments_family)
    empty_one_hot = np.zeros(num_instruments)

    main_path = "./musicnet/train_labels"
    train_labels = os.listdir(main_path)
    file_path = os.path.join(main_path, train_labels[index])
    file_name = os.path.splitext(os.path.basename(file_path))

    train_path = "./train_data_npy_100"

    csv_file = pd.read_csv(file_path)
    metadata_file = pd.read_csv("musicnet_metadata.csv")

    file_length = (
        int(metadata_file[metadata_file["id"] == int(file_name[0])]["seconds"]) + 1
    ) * 1000  # seconds

    # num_sequences = int(round(file_length / 10, 0) / 100)  # sequences

    num_sequences = len(
        os.listdir(os.path.join(train_path, os.listdir(train_path)[index]))
    )

    true_family = np.array([empty_one_hot_family for _ in range(num_sequences)])
    true_instruments = np.array([empty_one_hot for _ in range(num_sequences)])

    for record_index in range(csv_file.index.stop):
        record = csv_file.iloc[record_index]

        # [FAMILY]
        # get instrumental family
        for key, value in instruments_map_arr.items():
            if np.any(value == instruments[record.instrument]):
                record_instrument_genre = key  # instrumental family
                record_instrument = instruments[record.instrument]
                break

        # one hot encode one's family
        for key, value in instruments_group.items():
            if value == record_instrument_genre:
                record_instrument_genre_key = (
                    key  # position of one's family in instruments_group (index)
                )
                break

        # [INSTRUMENT]
        # one hot encode one's family
        for key, value in instruments.items():
            if value == record_instrument:
                record_instrument_key = (
                    key  # position of instrument in instruments (index)
                )
                break

        record_start_time = int(
            math.floor(convert(record.start_time) / 10) / 100
        )  # 1 sec
        record_end_time = int(math.floor(convert(record.end_time) / 10) / 100)

        for sequence in range(record_start_time, record_end_time):
            true_family[sequence][record_instrument_genre_key - 1] = 1
            true_instruments[sequence][record_instrument_key - 1] = 1

    return true_family, true_instruments


# one label for whole melspec
def prepare_labels_short(index):
    num_instruments_map = len(instruments_map_arr)
    empty_one_hot = np.zeros(num_instruments_map)

    main_path = "./musicnet/train_labels"
    train_labels = os.listdir(main_path)
    file_path = os.path.join(main_path, train_labels[index])

    csv_file = pd.read_csv(file_path)

    true_instruments_short = np.array(empty_one_hot)

    for record_index in range(csv_file.index.stop):
        record = csv_file.iloc[record_index]

        for key, value in instruments_map_arr.items():
            if np.any(value == instruments[record.instrument]):
                record_instrument_genre = key  # instrumental family
                break

        for key, value in instruments_group.items():
            if value == record_instrument_genre:
                record_instrument_genre_key = (
                    key  # position of one's family in instruments_group (index)
                )
                break

        true_instruments_short[record_instrument_genre_key - 1] = 1

    return true_instruments_short
