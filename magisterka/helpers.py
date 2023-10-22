def convert(data):
    value = int(round(data / 44100 * 1000, 0))
    return value
