import numpy as np
import torch
import torchaudio
import os
import matplotlib.pyplot as plt

from labels import prepare_labels
from helpers import convert

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


main_path = "./musicnet/test_data"
train_files = os.listdir(main_path)


sample_rate = 44100
hop_length = 441
n_fft = 1024
n_mels = 64
n_mfcc = 64
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

mfcc_transform = torchaudio.transforms.MFCC(
    sample_rate=sample_rate,
    n_mfcc=n_mfcc,
    melkwargs={
        "n_fft": n_fft,
        "hop_length": hop_length,
        "n_mels": n_mels,
        "mel_scale": "htk",
    },
)

# # visualize specific melspectrogram
# waveform, sample_rate = torchaudio.load("1727.wav")
# # mel_spectrogram = mel_transform(waveform)
# # mel_spectrogram_db = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)
# mfcc = mfcc_transform(waveform)
# mfcc_db = torchaudio.transforms.AmplitudeToDB()(mfcc)
# plt.figure(figsize=(12, 4))
# plt.imshow(mfcc_db[0].numpy(), origin="lower", aspect="auto")
# plt.colorbar(format="%+2.0f dB")
# plt.show()  # show specific melspec

# train_labels = []
# train_labels.append(prepare_labels(0))
# print(len(train_labels[0]))  # how many train labels

for index, filename in enumerate(train_files):
    file_path = os.path.join(main_path, train_files[index])
    waveform, sample_rate = torchaudio.load(file_path)

    for x in range(int(convert(waveform.shape[1]) / 10)):  # for 1/100 [sec]
        sample = 441  # 10 [ms]  = 1/100 [sec]
        start_sample = sample * x
        end_sample = sample * x + 513  # 513 for 10 [ms]
        if end_sample > waveform.shape[1]:
            break
        mel_spectrogram = mel_transform(waveform[:, start_sample:end_sample])
        mel_spectrogram_db = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)
        # mfcc = mfcc_transform(waveform[:, start_sample:end_sample])
        # mfcc_db = torchaudio.transforms.AmplitudeToDB()(mfcc)

        name = os.path.splitext(filename)[0] + f"_{x}" + ".npy"
        folder_path = "G:/data/test_data_npy/" + os.path.splitext(filename)[0]
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        path = "G:/data/test_data_npy/" + os.path.splitext(filename)[0] + f"/{name}"
        if not os.path.exists(path):
            np.save(path, mel_spectrogram_db)
