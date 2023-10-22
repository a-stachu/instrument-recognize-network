import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
import torch
from dataset import train_loader
from cnn import Net
from labels import prepare_labels_short
from sklearn.metrics import accuracy_score


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

    def training_step(self, batch):
        segments, labels = batch["segments"], batch["labels"]
        output = self(segments)
        # print(output.shape)
        # print(labels.shape)

        loss = torch.nn.MSELoss()
        loss = loss(output, labels)

        y_pred = []
        y_pred_final = []
        for sample in output:
            y_pred.append([1 if i >= 1e-01 else 0 for i in sample])
        for y in y_pred:
            if y_pred_final == []:
                y_pred_final = y
            else:
                for el in range(len(y)):
                    if y[el] == 1:
                        y_pred_final[el] = 1

        print("ACCURACY_WHOLE ->", accuracy_score(true_labels, y_pred_final))
        # argmax = torch.argmax(output, dim=1)
        # print(argmax, "argmax")

        # value, index = torch.topk(output, dim=1, k=12)
        # print(value, index)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def train_dataloader(self):
        return train_loader


true_labels = prepare_labels_short(0)

learner = Module(Net(12), true_labels)
checkpoint = pl.callbacks.ModelCheckpoint(monitor="loss")
trainer = pl.Trainer(
    accelerator="gpu", devices=1, max_epochs=20, callbacks=[checkpoint]
)
trainer.fit(learner, train_dataloaders=train_loader)
