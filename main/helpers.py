import math


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
