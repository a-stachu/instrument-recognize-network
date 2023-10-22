from instruments import *
import numpy as np

import torch
import torchaudio
import os
import matplotlib.pyplot as plt
from labels import prepare_labels
from helpers import convert

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

instruments_map = dict()

instruments_arr = np.array(list(instruments.values()))
instruments_group_arr = np.array(list(instruments_group.values()))

instruments_map_arr = {
    instruments_group_arr[0]: instruments_arr[0:16],  # keyboard OK
    instruments_group_arr[1]: instruments_arr[17:21],  # organ OK
    instruments_group_arr[2]: instruments_arr[22:24],  # accordion OK
    instruments_group_arr[3]: instruments_arr[25:32],  # guitar OK
    instruments_group_arr[4]: instruments_arr[33:40],  # bass OK
    instruments_group_arr[5]: instruments_arr[40:48],  # string OK
    instruments_group_arr[6]: instruments_arr[49:56],  # ensemble OK
    instruments_group_arr[7]: instruments_arr[57:64],  # brass OK
    instruments_group_arr[8]: instruments_arr[65:80],  # woodwind OK
    instruments_group_arr[9]: instruments_arr[81:104],  # synth OK
    instruments_group_arr[10]: instruments_arr[105:120],  # ethnic + percussion
    instruments_group_arr[11]: instruments_arr[121:128],  # sound effects
}

# print(instruments_map_arr)
main_path = "./musicnet/train_data"
train_files = os.listdir(main_path)
# file_path = os.path.join(main_path, train_files[0])


sample_rate = 44100
hop_length = 441
n_fft = 1024
n_mels = 64
f_min = 0
f_max = 22050

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=sample_rate,
    n_fft=n_fft,
    n_mels=n_mels,
    hop_length=hop_length,
    f_min=f_min,
    f_max=f_max,
)

# waveform, sample_rate = torchaudio.load("1759.wav")
# mel_spectrogram = mel_transform(waveform)
# mel_spectrogram_db = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)
# plt.figure(figsize=(12, 4))
# plt.imshow(mel_spectrogram_db[0].numpy(), cmap="viridis", origin="lower", aspect="auto")
# plt.colorbar(format="%+2.0f dB")
# plt.show()

# test_labels = []

# test_labels.append(prepare_labels(0))
# print(len(test_labels[0]))


# tworzenie melspec

# for index, filename in enumerate(train_files):
#     file_path = os.path.join(main_path, train_files[0])
#     waveform, sample_rate = torchaudio.load(file_path)

#     for x in range(int(convert(waveform.shape[1]) / 10)):
#         sample = 441 * 100
#         start_sample = sample * x
#         end_sample = sample * x + sample  # 513
#         if end_sample > waveform.shape[1]:
#             break
#         mel_spectrogram = mel_transform(waveform[:, start_sample:end_sample])
#         mel_spectrogram_db = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)
#         plt.figure(figsize=(12, 4))
#         plt.imshow(
#             mel_spectrogram_db[0].numpy(), cmap="viridis", origin="lower", aspect="auto"
#         )
#         plt.colorbar(format="%+2.0f dB")
#         plt.show()

#         name = os.path.splitext(filename)[0] + f"_{x}" + ".npy"
#         folder_path = "./train_data_npy_100/" + os.path.splitext(filename)[0]
#         if not os.path.exists(folder_path):
#             os.mkdir(folder_path)
#         path = "./train_data_npy_100/" + os.path.splitext(filename)[0] + f"/{name}"
#         if not os.path.exists(path):
#             np.save(path, mel_spectrogram_db)
