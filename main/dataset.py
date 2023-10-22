import torch
import os
import numpy as np
from labels import prepare_labels


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


train_path = "./train_data_npy_100/1727"
train_files_path = os.listdir(train_path)
train_files = []
train_files_labels = []

test_path = "./train_data_npy_100/1728"
test_files_path = os.listdir(test_path)
test_files = []
test_files_labels = []

train_dimension = 0  # init state
test_dimension = 0  # init state

# train files
for file in range(len(train_files_path)):
    file_path = os.path.join(train_path, train_files_path[file])
    melspectrogram = np.load(file_path)
    train_files.append(melspectrogram)

# test files
for file in range(len(test_files_path)):
    file_path = os.path.join(test_path, test_files_path[file])
    melspectrogram = np.load(file_path)
    test_files.append(melspectrogram)

train_files_labels = prepare_labels(0)[0 : len(train_files)]
test_files_labels = prepare_labels(1)[0 : len(test_files)]

# all segments of melspec have to be the same size
for file in train_files:
    dim = torch.Tensor(file).size(dim=2)
    if train_dimension < dim:
        train_dimension = dim

for file in test_files:
    dim = torch.Tensor(file).size(dim=2)
    if test_dimension < dim:
        test_dimension = dim

train_dataset = Dataset(train_files, train_files_labels, train_dimension)
test_dataset = Dataset(test_files, test_files_labels, test_dimension)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)
