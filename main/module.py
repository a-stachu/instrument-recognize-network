import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
import torch
from dataset import train_loader, test_loader
from cnn import Net
from labels import prepare_labels_short
from sklearn.metrics import accuracy_score
from pytorch_lightning.loggers import TensorBoardLogger


class Module(pl.LightningModule):
    def __init__(
        self,
        model,
        true_labels,
        learning_rate=1e-3,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.model = model
        self.true_labels = true_labels

    def forward(self, x):
        return self.model(x)

    def step(self, x, y):
        prediction = self(x)
        loss = torch.nn.MSELoss()
        loss = loss(prediction, y)  # y -> true labels

        y_pred = []
        y_pred_all = []  # final array with all predictions as a whole
        for sample in prediction:
            y_pred.append([1 if i >= 1e-01 else 0 for i in sample])
        for y in y_pred:
            if y_pred_all == []:  # initial value
                y_pred_all = y
            else:
                for el in range(
                    len(y)
                ):  # if one of the values is equal to 1 add it to the array eg. new array: [1 0 0 1] | old array: [1 1 0 0] | result: [1 1 0 1]
                    if y[el] == 1:
                        y_pred_all[el] = 1

        accuracy = accuracy_score(true_labels, y_pred_all)

        # argmax = torch.argmax(output, dim=1)
        # value, index = torch.topk(output, dim=1, k=12)

        return loss, accuracy

    def training_step(self, batch):
        segments, labels = batch["segments"], batch["labels"]
        loss, accuracy = self.step(segments, labels)
        self.log("accuracy", accuracy, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        segments, labels = batch["segments"], batch["labels"]
        loss, accuracy = self.step(segments, labels)
        self.log("accuracy", accuracy, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def train_dataloader(self):
        return train_loader

    def test_dataloader(self):
        return test_loader


logger = TensorBoardLogger("logs/", name="logger")
true_labels = prepare_labels_short(0)

learner = Module(Net(12), true_labels)
checkpoint = pl.callbacks.ModelCheckpoint(monitor="loss")
trainer = pl.Trainer(
    accelerator="gpu", devices=1, max_epochs=100, callbacks=[checkpoint], logger=logger
)
trainer.fit(learner, train_dataloaders=train_loader)

# trainer.test(model=learner, dataloaders=test_loader)
