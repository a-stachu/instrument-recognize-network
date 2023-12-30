import pandas as pd
import os
import numpy as np

# import math

from instruments import (
    instruments_map_arr,
    instruments_map_arr_short,
    instruments_map_arr_alternative_short,
    instruments,
    instruments_group,
    instruments_short,
)

# from instruments import instruments_map_arr, instruments, instruments_group

from helpers import *


######## 2 case
def shorten(input, short_set):
    for key, value in short_set.items():
        if isinstance(value, np.ndarray):
            for val in value:
                if input == val:
                    return key
        else:
            if input == value:
                return key


# one label for one segment of melspec
def prepare_labels(index, case, train_path):
    num_instruments_family = len(instruments_map_arr)
    num_instruments = len(instruments)
    empty_one_hot_family = np.zeros(num_instruments_family)
    empty_one_hot = np.zeros(num_instruments)

    empty_one_hot_family_short = np.zeros(len(instruments_map_arr_short))
    empty_one_hot_short = np.zeros(len(instruments_short))

    main_path = "./musicnet/train_labels"
    train_labels = os.listdir(main_path)
    file_path = os.path.join(main_path, train_labels[index])
    csv_file = pd.read_csv(file_path)

    num_sequences = len(
        os.listdir(os.path.join(train_path, os.listdir(train_path)[index]))
    )

    # print(num_sequences)

    true_family = np.array([empty_one_hot_family for _ in range(num_sequences)])
    true_instruments = np.array([empty_one_hot for _ in range(num_sequences)])

    true_family_short = np.array(
        [empty_one_hot_family_short for _ in range(num_sequences)]
    )
    true_instruments_short = np.array(
        [empty_one_hot_short for _ in range(num_sequences)]
    )

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

        record_start_time = count_record_time(record.start_time, case)
        record_end_time = count_record_time(record.end_time, case)

        for sequence in range(record_start_time, record_end_time):
            # print(
            #     sequence,
            #     record_instrument_genre_key,
            #     record_instrument_key,
            #     record_start_time,
            #     record_end_time,
            # )

            true_family[sequence][record_instrument_genre_key - 1] = 1
            true_family_short[sequence][
                shorten(
                    record_instrument_genre_key, instruments_map_arr_alternative_short
                )
                - 1
            ] = 1
            true_instruments[sequence][record_instrument_key - 1] = 1
            true_instruments_short[sequence][
                shorten(record_instrument_key, instruments_short) - 1
            ] = 1

    return true_family_short, true_instruments_short


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

    true_family_short = np.array(np.zeros(len(instruments_map_arr_short)))
    true_family = np.array(empty_one_hot_family)
    true_instruments = np.array(empty_one_hot)
    true_instruments_short = np.array(np.zeros(len(instruments_short)))

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

        true_family[record_instrument_genre_key - 1] = 1
        true_instruments[record_instrument_key - 1] = 1

        ########## 1 case
        keys = []
        for index in range(len(true_instruments)):
            if true_instruments[index] == 1:
                keys.append(index + 1)
        keys = list(set(keys))

        for key, value in instruments_short.items():
            # print(value, keys, key)
            if value in keys:
                true_instruments_short[key - 1] = 1

        #####

        keys2 = []
        for index in range(len(true_family)):
            if true_family[index] == 1:
                keys2.append(index + 1)
        keys2 = list(set(keys2))

        for key, value in instruments_map_arr_alternative_short.items():
            for val in value:
                if val in keys2:
                    true_family_short[key] = 1

    return true_family_short, true_instruments_short
