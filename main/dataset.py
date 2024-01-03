import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self, melspectrograms, mfcc, labels_instruments, labels_family, dimension
    ):
        self.melspectrograms = melspectrograms
        self.mfcc = mfcc
        self.labels_instruments = labels_instruments
        self.labels_family = labels_family
        self.dimension = dimension

    def __len__(self):  # optional to check if dimensions are right for labels
        return len(self.melspectrograms)

    def __getitem__(self, index):
        mel_spec = self.melspectrograms[index]
        mfcc = self.mfcc[index]
        label_instruments = self.labels_instruments[index]
        label_family = self.labels_family[index]
        dimension = self.dimension

        if torch.Tensor(mel_spec).size(dim=2) < dimension:
            padding = dimension - torch.Tensor(mel_spec).size(dim=2)
            resulting_tensor = torch.cat(
                (torch.Tensor(mel_spec), torch.zeros(1, 64, padding)), dim=2
            )
        else:
            resulting_tensor = torch.Tensor(mel_spec)

        if torch.Tensor(mfcc).size(dim=2) < dimension:
            padding = dimension - torch.Tensor(mfcc).size(dim=2)
            resulting_tensor_mfcc = torch.cat(
                (torch.Tensor(mfcc), torch.zeros(1, 64, padding)), dim=2
            )
        else:
            resulting_tensor_mfcc = torch.Tensor(mfcc)

        sample = {
            "segments": resulting_tensor,
            "segments_mfcc": resulting_tensor_mfcc,
            "labels_instruments": torch.Tensor(label_instruments),
            "labels_family": torch.Tensor(label_family),
        }
        return sample


train_files = []
train_files_labels = []
test_files = []
test_files_labels = []

train_files_mfcc = []
train_files_labels_mfcc = []
test_files_mfcc = []
test_files_labels_mfcc = []

train_dimension = 0  # init state
test_dimension = 0  # init state

# all segments of melspec and mfcc have to be the same size
for file in train_files:
    dim = torch.Tensor(file).size(dim=2)
    if train_dimension < dim:
        train_dimension = dim

for file in test_files:
    dim = torch.Tensor(file).size(dim=2)
    if test_dimension < dim:
        test_dimension = dim

for file in train_files_mfcc:
    dim = torch.Tensor(file).size(dim=2)
    if train_dimension < dim:
        train_dimension = dim

for file in test_files_mfcc:
    dim = torch.Tensor(file).size(dim=2)
    if test_dimension < dim:
        test_dimension = dim
