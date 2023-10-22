import numpy as np

import torchaudio
import matplotlib.pyplot as plt
import torch
from labels import prepare_labels, prepare_labels_short
from helpers import convert

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

waveform, sample_rate = torchaudio.load("1727.wav")
print(convert(waveform.shape[1]) / 10)

mel_spectrogram = mel_transform(waveform[:, 0 : 441 * 7 + 513])
mel_spectrogram_db = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)
plt.figure(figsize=(12, 4))
plt.imshow(mel_spectrogram_db[0].numpy(), cmap="plasma", origin="lower", aspect="auto")
plt.colorbar(format="%+2.0f dB")
plt.show()

dim = 106904

padding = dim - mel_spectrogram_db.size(dim=2)
zeros = torch.zeros(1, 64, padding)
resulting_tensor = torch.cat((mel_spectrogram_db, torch.zeros(1, 64, padding)), dim=2)
print(resulting_tensor.shape)
# test_labels = []
# test_labels.append(prepare_labels_short(0))
test_labels = prepare_labels(0)
print(len(test_labels))
