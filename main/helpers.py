import math
import os


def convert(data):
    value = int(round(data / 44100 * 1000, 0))
    return value


def count_record_time(data, case):
    if case == 1000:  # 1 [sec]
        sample = 100
    if case == 100:  # 100 [ms]
        sample = 10
    if case == 10:  # 10 [ms]
        sample = 1

    value = int(math.floor(convert(data) / 10) / sample)
    return value


def populate_table(tensor_true, tensor_predicted, data, instruments, expanded=None):
    for i in range(len(tensor_true)):
        for j in range(len(tensor_true[i])):
            if tensor_true[i][j] == 1:
                if 1 >= tensor_predicted[i][j] >= 0.5:
                    if expanded:
                        data[instruments[expanded[j + 1]]] += 1
                    else:
                        data[instruments[j + 1]] += 1


def list_files_in_subdirectories(directory):
    file_list = []

    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_list.append(file_path)

    return file_list
