import torch
import os
import numpy as np
from labels import prepare_labels_short, prepare_labels


class Dataset(torch.utils.data.Dataset):
    def __init__(self, melspectrograms, labels, dimension):
        self.melspectrograms = melspectrograms
        self.labels = labels
        self.dimension = dimension

    def __len__(self):
        return len(self.melspectrograms)

    def __getitem__(self, index):
        mel_spec = self.melspectrograms[index]
        label = self.labels[index]
        dimension = self.dimension

        # print("len", segment_length_samples)
        # print(len(label))
        # print(type(mel_spec))
        # print(torch.Tensor(mel_spec).shape)
        # print("test", len(self.melspectrograms[index]))

        # print(1 * segment_length_samples)
        # print(2 * segment_length_samples)
        # print(torch.Tensor(mel_spec).)

        # for i in range(segment_length_samples):
        #     start_sample = i * segment_length_samples

        #     end_sample = (i + 1) * segment_length_samples

        #     segment = mel_spec[:, start_sample:end_sample]
        #     # print(segment, "segment")
        #     segments.append(segment)
        #     segment_labels.append(label[i])
        if torch.Tensor(mel_spec).size(dim=2) < dimension:
            padding = dimension - torch.Tensor(mel_spec).size(dim=2)
            resulting_tensor = torch.cat(
                (torch.Tensor(mel_spec), torch.zeros(1, 64, padding)), dim=2
            )
        else:
            resulting_tensor = torch.Tensor(mel_spec)

        sample = {
            "segments": resulting_tensor,
            "labels": torch.Tensor(label),
        }
        # print(resulting_tensor.shape, "segments")
        # print(torch.Tensor(label).shape)
        return sample


main_path = "./train_data_npy_100/1727"
train_files_path = os.listdir(main_path)
train_files = []
train_files_labels = []
dimension = 0

for file in range(len(train_files_path)):
    file_path = os.path.join(main_path, train_files_path[file])
    melspectrogram = np.load(file_path)
    train_files.append(melspectrogram)

# for file in range(len(train_files)):
#     # label = prepare_labels_short(file)
#     label = prepare_labels(0)  #
#     train_files_labels.append(label)

train_files_labels = prepare_labels(0)[0 : len(train_files)]

for file in train_files:
    dim = torch.Tensor(file).size(dim=2)
    if dimension < dim:
        dimension = dim

train_dataset = Dataset(train_files, train_files_labels, dimension)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
