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
    train_path = "./train_data_npy_100"
    train_labels = os.listdir(main_path)
    file_path = os.path.join(main_path, train_labels[index])
    csv_file = pd.read_csv(file_path)

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
                record_instrument = instruments[record.instrument]  # instrument
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
    num_instruments_family = len(instruments_map_arr)
    num_instruments = len(instruments)
    empty_one_hot_family = np.zeros(num_instruments_family)
    empty_one_hot = np.zeros(num_instruments)

    main_path = "./musicnet/train_labels"
    train_labels = os.listdir(main_path)
    file_path = os.path.join(main_path, train_labels[index])

    csv_file = pd.read_csv(file_path)

    true_family_short = np.array(empty_one_hot_family)
    true_instruments_short = np.array(empty_one_hot)

    for record_index in range(csv_file.index.stop):
        record = csv_file.iloc[record_index]

        # [FAMILY]
        # get instrumental family
        for key, value in instruments_map_arr.items():
            if np.any(value == instruments[record.instrument]):
                record_instrument_genre = key  # instrumental family
                record_instrument = instruments[record.instrument]  # instrument
                break

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

        true_family_short[record_instrument_genre_key - 1] = 1
        true_instruments_short[record_instrument_key - 1] = 1

    return true_family_short, true_instruments_short
