import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
import torch
import torchmetrics
from torchmetrics.classification import (
    ConfusionMatrix,
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
    BinaryAccuracy,
)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
from instruments import (
    instruments_map_arr_alternative_short as instruments_map_arr_alternative,
)


class Module(pl.LightningModule):
    def __init__(
        self,
        model_instruments,
        model_family,
        true_labels_instruments,
        true_labels_family,
        learning_rate=1e-3,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.model_instruments = model_instruments
        self.model_family = model_family
        self.true_labels_instruments = true_labels_instruments
        self.true_labels_family = true_labels_family

        self.training_step_preds_instruments = []
        self.training_step_target_instruments = []
        self.training_step_preds_family = []
        self.training_step_target_family = []

    def forward(self, x):
        output_instruments = self.model_instruments(x)
        output_family = self.model_family(x)
        return output_instruments, output_family

    def step(self, x, y_1, y_2):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        output_instruments, output_family = self(x)
        prediction_2 = output_instruments
        prediction_1 = output_family
        loss = torch.nn.MSELoss()
        loss_family = loss(prediction_1, y_1)  # y -> true labels
        loss_instruments = loss(prediction_2, y_2)  # y -> true labels

        precision_calc = BinaryPrecision(threshold=1e-04)
        precision_calc.to(device)
        precision_family = precision_calc(prediction_1, y_1).item()
        precision_instruments = precision_calc(prediction_2, y_2).item()

        recall_calc = BinaryRecall(threshold=1e-04)
        recall_calc.to(device)
        recall_family = recall_calc(prediction_1, y_1).item()
        recall_instruments = recall_calc(prediction_2, y_2).item()

        f1_calc = BinaryF1Score(threshold=1e-04)
        f1_calc.to(device)
        f1_family = f1_calc(prediction_1, y_1).item()
        f1_instruments = f1_calc(prediction_2, y_2).item()

        y_pred = []
        y_pred_all = []  # final array with all predictions as a whole
        for sample in prediction_1:
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

        y_pred_2, y_pred_1 = self.score_level_fusion(prediction_1, prediction_2)

        accuracy_calc = BinaryAccuracy(threshold=1e-04)
        accuracy_calc.to(device)
        # print(y_pred_2, y_2)
        accuracy = accuracy_calc(torch.Tensor(y_pred_2).to(device), y_2)
        accuracy_family = accuracy_calc(torch.Tensor(prediction_1).to(device), y_1)
        accuracy_instrument = accuracy_calc(torch.Tensor(prediction_2).to(device), y_2)

        return (
            loss_family,
            loss_instruments,
            accuracy,
            accuracy_family,
            accuracy_instrument,
            precision_family,
            precision_instruments,
            recall_family,
            recall_instruments,
            f1_family,
            f1_instruments,
        )

    def training_step(self, batch):
        segments, labels_family, labels_instruments = (
            batch["segments"],
            batch["labels_family"],
            batch["labels_instruments"],
        )
        (
            loss_family,
            loss_instruments,
            accuracy,
            accuracy_family,
            accuracy_instrument,
            precision_family,
            precision_instruments,
            recall_family,
            recall_instruments,
            f1_family,
            f1_instruments,
        ) = self.step(segments, labels_family, labels_instruments)

        # self.training_step_preds.append(prediction)
        # self.training_step_target.append(labels)

        self.log("accuracy", accuracy, prog_bar=True)
        self.log("accuracy_instrument", accuracy_instrument, prog_bar=True)
        self.log("accuracy_family", accuracy_family, prog_bar=True)
        self.log("precision_family", precision_family, prog_bar=True)
        self.log("precision_instruments", precision_instruments, prog_bar=True)
        self.log("recall_family", recall_family, prog_bar=True)
        self.log("recall_instruments", recall_instruments, prog_bar=True)
        self.log("F1_family", f1_family, prog_bar=True)
        self.log("F1_instruments", f1_instruments, prog_bar=True)
        return {
            "loss": loss_family + loss_instruments,
        }

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
        self.training_step_preds_family.clear()  # free memory
        self.training_step_target_instruments.clear()  # free memory
        self.training_step_preds_instruments.clear()  # free memory
        self.training_step_target_family.clear()  # free memory

    def score_level_fusion(self, prediction_family, prediction_instruments):
        y_pred_1 = []

        prediction_1 = prediction_family
        for sample in prediction_1:
            y_pred_1.append([1 if i >= 1e-01 else 0 for i in sample])

        y_pred_2 = []

        prediction_2 = prediction_instruments

        decoded_y_pred_1 = []
        for prediction in y_pred_1:
            boosted_indexes_partial = []
            for i, j in enumerate(prediction):
                if j == 1:
                    boosted_indexes_partial.append(i)
            decoded_y_pred_1.append(
                np.unique(np.array(boosted_indexes_partial).flatten())
            )

        boosted_instruments = []

        for decoded_prediction in decoded_y_pred_1:
            boosted_instruments_partial = []
            for key, value in instruments_map_arr_alternative.items():
                if np.isin(decoded_prediction, key).any():
                    boosted_instruments_partial.append(value)
                else:
                    continue

            if np.array(boosted_instruments_partial).size > 0:
                if len(boosted_instruments_partial) == 1:
                    boosted_instruments.append(np.array(boosted_instruments_partial))
                else:
                    try:
                        result = np.concatenate(
                            (np.array(boosted_instruments_partial)).flatten()
                        )
                    except ValueError as e:
                        if "zero-dimensional" in str(e):
                            correct_result = (
                                np.array(boosted_instruments_partial)
                            ).flatten()
                            boosted_instruments.append(np.unique(correct_result))
                    else:
                        boosted_instruments.append(np.unique(result))
            else:
                boosted_instruments.append([])

        for index in range(len(prediction_2)):
            predicted_instruments = prediction_2[index].cpu().detach().numpy()
            predicted_instruments_partial = []
            threshold = 0

            for i in range(len(predicted_instruments)):
                try:
                    custom_range = boosted_instruments[index]

                except IndexError as e:
                    custom_range = boosted_instruments[index - 1]

                    if i in custom_range:
                        threshold == 1e-04
                    else:
                        threshold == 1
                    predicted_instruments_partial.append(
                        1 if predicted_instruments[i] >= threshold else 0
                    )

                else:
                    if i in custom_range:
                        threshold == 1e-04
                    else:
                        threshold == 1
                    predicted_instruments_partial.append(
                        1 if predicted_instruments[i] >= threshold else 0
                    )

                if len(predicted_instruments_partial) == 11:
                    y_pred_2.append(predicted_instruments_partial)

        return y_pred_2, y_pred_1

    # def accuracy_whole(self):
