import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
import torch
from dataset import train_loader, test_loader
from cnn import Net
from labels import prepare_labels_short
from sklearn.metrics import accuracy_score
from pytorch_lightning.loggers import TensorBoardLogger
import torchmetrics
from torchmetrics.classification import (
    ConfusionMatrix,
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np


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

        self.training_step_preds = []
        self.training_step_target = []

    def forward(self, x):
        return self.model(x)

    def step(self, x, y):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        prediction = self(x)
        loss = torch.nn.MSELoss()
        loss = loss(prediction, y)  # y -> true labels

        precision_calc = BinaryPrecision(threshold=1e-04)
        precision_calc.to(device)
        precision = precision_calc(prediction, y).item()

        recall_calc = BinaryRecall(threshold=1e-04)
        recall_calc.to(device)
        recall = recall_calc(prediction, y).item()

        f1_calc = BinaryF1Score(threshold=1e-04)
        f1_calc.to(device)
        f1 = f1_calc(prediction, y).item()

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

        accuracy = 0.1  # accuracy_score(y, prediction)

        # argmax = torch.argmax(output, dim=1)
        # value, index = torch.topk(output, dim=1, k=12)

        return loss, accuracy, prediction, precision, recall, f1

    def training_step(self, batch):
        segments, labels = batch["segments"], batch["labels"]
        loss, accuracy, prediction, precision, recall, f1 = self.step(segments, labels)
        self.training_step_preds.append(prediction)
        self.training_step_target.append(labels)
        self.log("accuracy", accuracy, prog_bar=True)
        self.log("precision", precision, prog_bar=True)
        self.log("recall", recall, prog_bar=True)
        self.log("F1", f1, prog_bar=True)
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

    def on_train_epoch_end(self):
        # [CONFUSION MATRIX skeleton]
        # preds = torch.cat(self.training_step_preds)
        # target = torch.cat(self.training_step_target)
        target = torch.tensor([[0, 1, 0], [1, 0, 1]])
        preds = torch.tensor([[0, 0, 1], [1, 0, 1]])
        confusion_matrix = torchmetrics.ConfusionMatrix(
            task="multiclass", num_classes=2, threshold=0.05
        )
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        # confusion_matrix.to(device)
        confusion_matrix(preds, target)
        cm = confusion_matrix.compute().detach().cpu().numpy()
        confusion_matrix_computed = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

        df_cm = pd.DataFrame(
            confusion_matrix_computed,
        )

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.subplots_adjust(left=0.05, right=0.65)
        sn.set(font_scale=1.2)
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt=".2f", ax=ax)

        # plt.show()

        self.training_step_preds.clear()  # free memory
        self.training_step_target.clear()  # free memory


logger = TensorBoardLogger("logs/", name="logger")
true_labels = prepare_labels_short(0)

learner = Module(Net(12), true_labels)
checkpoint = pl.callbacks.ModelCheckpoint(monitor="loss")
trainer = pl.Trainer(
    accelerator="gpu", devices=1, max_epochs=100, callbacks=[checkpoint], logger=logger
)
trainer.fit(learner, train_dataloaders=train_loader)

# trainer.test(model=learner, dataloaders=test_loader)
