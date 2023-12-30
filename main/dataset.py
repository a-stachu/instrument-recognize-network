import torch
import os
import numpy as np

# from labels import prepare_labels


class Dataset(torch.utils.data.Dataset):
    def __init__(self, melspectrograms, labels_instruments, labels_family, dimension):
        self.melspectrograms = melspectrograms
        self.labels_instruments = labels_instruments
        self.labels_family = labels_family
        self.dimension = dimension

    def __len__(self):
        return len(self.melspectrograms)

    def __getitem__(self, index):
        mel_spec = self.melspectrograms[index]
        label_instruments = self.labels_instruments[index]
        label_family = self.labels_family[index]
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
            "labels_instruments": torch.Tensor(label_instruments),
            "labels_family": torch.Tensor(label_family),
        }
        # print(resulting_tensor.shape, "segments")
        # print(torch.Tensor(label).shape)
        return sample


file_count = 0
for root, dirs, files in os.walk("./train_data_npy_100"):
    file_count += len(files)
# print(file_count)


train_files = []
train_files_labels = []

test_files = []
test_files_labels = []

train_dimension = 0  # init state
test_dimension = 0  # init state


# test_files_labels = prepare_labels(1, 1000)[1]

# print(train_files_labels_instruments)

# all segments of melspec have to be the same size
for file in train_files:
    dim = torch.Tensor(file).size(dim=2)
    if train_dimension < dim:
        train_dimension = dim

for file in test_files:
    dim = torch.Tensor(file).size(dim=2)
    if test_dimension < dim:
        test_dimension = dim

# test_dataset = Dataset(test_files, test_files_labels, test_dimension)


# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)
