import os


def convert(data):
    value = int(round(data / 44100 * 1000, 0))
    return value


def list_files_in_subdirectories(directory):
    file_list = []

    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_list.append(file_path)

    return file_list
