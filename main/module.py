import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
import torch

from torchmetrics.classification import (
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
    BinaryAccuracy,
)
import pandas as pd

import numpy as np
from instruments import (
    instruments_map_arr_alternative_short as instruments_map_arr_alternative,
    instruments_group_short,
    instruments_short,
    instruments,
)

from loaders import load_test_data, load_training_data
from helpers import populate_table


class Module(pl.LightningModule):
    def __init__(
        self,
        model_instruments,
        model_family,
        true_labels_instruments,
        true_labels_family,
        variant,
        learning_rate=1e-3,
    ):
        super().__init__()

        self.learning_rate = learning_rate

        self.model_instruments = model_instruments
        self.model_family = model_family
        self.true_labels_instruments = true_labels_instruments
        self.true_labels_family = true_labels_family

        self.variant = variant

        self.training_step_preds_instruments = {}
        self.training_step_target_instruments = {}
        self.training_step_preds_family = {}
        self.training_step_target_family = {}

        self.training_step_preds_instruments_mfcc = {}
        self.training_step_target_instruments_mfcc = {}
        self.training_step_preds_family_mfcc = {}
        self.training_step_target_family_mfcc = {}

    def forward(self, x):
        output_instruments = self.model_instruments(x)
        output_family = self.model_family(x)
        return output_instruments, output_family

    def step(self, x1, x2, y_1, y_2):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        output_instruments, output_family = self(x1)
        output_instruments_mfcc, output_family_mfcc = self(x2)
        prediction_2 = output_instruments
        prediction_1 = output_family
        prediction_2_mfcc = output_instruments_mfcc
        prediction_1_mfcc = output_family_mfcc
        loss = torch.nn.MSELoss()
        loss_family = loss(prediction_1, y_1)  # y -> true labels
        loss_instruments = loss(prediction_2, y_2)  # y -> true labels
        loss_family_mfcc = loss(prediction_1_mfcc, y_1)  # y -> true labels
        loss_instruments_mfcc = loss(prediction_2_mfcc, y_2)  # y -> true labels

        precision_calc = BinaryPrecision(threshold=1e-04)
        precision_calc.to(device)
        precision_family = precision_calc(prediction_1, y_1).item()
        precision_instruments = precision_calc(prediction_2, y_2).item()
        precision_family_mfcc = precision_calc(prediction_1_mfcc, y_1).item()
        precision_instruments_mfcc = precision_calc(prediction_2_mfcc, y_2).item()

        recall_calc = BinaryRecall(threshold=1e-04)
        recall_calc.to(device)
        recall_family = recall_calc(prediction_1, y_1).item()
        recall_instruments = recall_calc(prediction_2, y_2).item()
        recall_family_mfcc = recall_calc(prediction_1_mfcc, y_1).item()
        recall_instruments_mfcc = recall_calc(prediction_2_mfcc, y_2).item()

        f1_calc = BinaryF1Score(threshold=1e-04)
        f1_calc.to(device)
        f1_family = f1_calc(prediction_1, y_1).item()
        f1_instruments = f1_calc(prediction_2, y_2).item()
        f1_family_mfcc = f1_calc(prediction_1_mfcc, y_1).item()
        f1_instruments_mfcc = f1_calc(prediction_2_mfcc, y_2).item()

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

        y_pred_2, y_pred_2_mfcc, y_pred_2_combined = self.score_level_fusion(
            prediction_1, prediction_2, prediction_1_mfcc, prediction_2_mfcc
        )

        accuracy_calc = BinaryAccuracy(threshold=1e-04)
        accuracy_calc.to(device)

        accuracy = accuracy_calc(torch.Tensor(y_pred_2).to(device), y_2)
        accuracy_family = accuracy_calc(torch.Tensor(prediction_1).to(device), y_1)
        accuracy_instrument = accuracy_calc(torch.Tensor(prediction_2).to(device), y_2)

        accuracy_mfcc = accuracy_calc(torch.Tensor(y_pred_2_mfcc).to(device), y_2)
        accuracy_family_mfcc = accuracy_calc(
            torch.Tensor(prediction_1_mfcc).to(device), y_1
        )
        accuracy_instrument_mfcc = accuracy_calc(
            torch.Tensor(prediction_2_mfcc).to(device), y_2
        )

        accuracy_combined = accuracy_calc(
            torch.Tensor(y_pred_2_combined).to(device), y_2
        )

        for value in instruments_group_short.values():
            self.training_step_preds_family[value] = 0
            self.training_step_target_family[value] = 0
            self.training_step_preds_family_mfcc[value] = 0
            self.training_step_target_family_mfcc[value] = 0

        for value in instruments_short.values():
            value = instruments[value]
            self.training_step_preds_instruments[value] = 0
            self.training_step_target_instruments[value] = 0
            self.training_step_preds_instruments_mfcc[value] = 0
            self.training_step_target_instruments_mfcc[value] = 0

        tensor_true1 = y_1.cpu().numpy()
        tensor_predicted1 = prediction_1.cpu().detach().numpy()
        tensor_true2 = y_2.cpu().numpy()
        tensor_predicted2 = prediction_2.cpu().detach().numpy()

        tensor_predicted1_mfcc = prediction_1_mfcc.cpu().detach().numpy()
        tensor_predicted2_mfcc = prediction_2_mfcc.cpu().detach().numpy()

        populate_table(
            tensor_true1,
            tensor_predicted1_mfcc,
            self.training_step_preds_family_mfcc,
            instruments_group_short,
        )
        populate_table(
            tensor_true1,
            tensor_predicted1_mfcc,
            self.training_step_target_family_mfcc,
            instruments_group_short,
        )

        populate_table(
            tensor_true2,
            tensor_predicted2_mfcc,
            self.training_step_preds_instruments_mfcc,
            instruments,
            instruments_short,
        )
        populate_table(
            tensor_true2,
            tensor_predicted2_mfcc,
            self.training_step_target_instruments_mfcc,
            instruments,
            instruments_short,
        )

        populate_table(
            tensor_true1,
            tensor_predicted1,
            self.training_step_preds_family,
            instruments_group_short,
        )
        populate_table(
            tensor_true1,
            tensor_predicted1,
            self.training_step_target_family,
            instruments_group_short,
        )

        populate_table(
            tensor_true2,
            tensor_predicted2,
            self.training_step_preds_instruments,
            instruments,
            instruments_short,
        )
        populate_table(
            tensor_true2,
            tensor_predicted2,
            self.training_step_target_instruments,
            instruments,
            instruments_short,
        )

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
            loss_family_mfcc,
            loss_instruments_mfcc,
            accuracy_mfcc,
            accuracy_family_mfcc,
            accuracy_instrument_mfcc,
            precision_family_mfcc,
            precision_instruments_mfcc,
            recall_family_mfcc,
            recall_instruments_mfcc,
            f1_family_mfcc,
            f1_instruments_mfcc,
            accuracy_combined,
        )

    def training_step(self, batch):
        segments, segments_mfcc, labels_family, labels_instruments = (
            batch["segments"],
            batch["segments_mfcc"],
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
            loss_family_mfcc,
            loss_instruments_mfcc,
            accuracy_mfcc,
            accuracy_family_mfcc,
            accuracy_instrument_mfcc,
            precision_family_mfcc,
            precision_instruments_mfcc,
            recall_family_mfcc,
            recall_instruments_mfcc,
            f1_family_mfcc,
            f1_instruments_mfcc,
            accuracy_combined,
        ) = self.step(segments, segments_mfcc, labels_family, labels_instruments)

        self.log("accuracy", accuracy, prog_bar=True)
        # self.log("accuracy_instrument", accuracy_instrument, prog_bar=True)
        # self.log("accuracy_family", accuracy_family, prog_bar=True)
        # self.log("precision_family", precision_family, prog_bar=True)
        # self.log("precision_instruments", precision_instruments, prog_bar=True)
        # self.log("recall_family", recall_family, prog_bar=True)
        # self.log("recall_instruments", recall_instruments, prog_bar=True)
        # self.log("F1_family", f1_family, prog_bar=True)
        # self.log("F1_instruments", f1_instruments, prog_bar=True)

        self.log("accuracy_mfcc", accuracy_mfcc, prog_bar=True)
        # self.log("accuracy_instrument_mfcc", accuracy_instrument_mfcc, prog_bar=True)
        # self.log("accuracy_family_mfcc", accuracy_family_mfcc, prog_bar=True)
        # self.log("precision_family_mfcc", precision_family_mfcc, prog_bar=True)
        # self.log(
        #     "precision_instruments_mfcc", precision_instruments_mfcc, prog_bar=True
        # )
        # self.log("recall_family_mfcc", recall_family_mfcc, prog_bar=True)
        # self.log("recall_instruments_mfcc", recall_instruments_mfcc, prog_bar=True)
        # self.log("F1_family_mfcc", f1_family_mfcc, prog_bar=True)
        # self.log("F1_instruments_mfcc", f1_instruments_mfcc, prog_bar=True)
        self.log("accuracy_combined", accuracy_combined, prog_bar=True)
        return {
            "loss": loss_family
            + loss_instruments
            + loss_family_mfcc
            + loss_instruments_mfcc,
        }

    def test_step(self, batch, batch_idx):
        segments, segments_mfcc, labels_family, labels_instruments = (
            batch["segments"],
            batch["segments_mfcc"],
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
            loss_family_mfcc,
            loss_instruments_mfcc,
            accuracy_mfcc,
            accuracy_family_mfcc,
            accuracy_instrument_mfcc,
            precision_family_mfcc,
            precision_instruments_mfcc,
            recall_family_mfcc,
            recall_instruments_mfcc,
            f1_family_mfcc,
            f1_instruments_mfcc,
            accuracy_combined,
        ) = self.step(segments, segments_mfcc, labels_family, labels_instruments)

        self.log("accuracy", accuracy, prog_bar=True)
        # self.log("accuracy_instrument", accuracy_instrument, prog_bar=True)
        # self.log("accuracy_family", accuracy_family, prog_bar=True)
        # self.log("precision_family", precision_family, prog_bar=True)
        # self.log("precision_instruments", precision_instruments, prog_bar=True)
        # self.log("recall_family", recall_family, prog_bar=True)
        # self.log("recall_instruments", recall_instruments, prog_bar=True)
        # self.log("F1_family", f1_family, prog_bar=True)
        # self.log("F1_instruments", f1_instruments, prog_bar=True)

        self.log("accuracy_mfcc", accuracy_mfcc, prog_bar=True)
        # self.log("accuracy_instrument_mfcc", accuracy_instrument_mfcc, prog_bar=True)
        # self.log("accuracy_family_mfcc", accuracy_family_mfcc, prog_bar=True)
        # self.log("precision_family_mfcc", precision_family_mfcc, prog_bar=True)
        # self.log(
        #     "precision_instruments_mfcc", precision_instruments_mfcc, prog_bar=True
        # )
        # self.log("recall_family_mfcc", recall_family_mfcc, prog_bar=True)
        # self.log("recall_instruments_mfcc", recall_instruments_mfcc, prog_bar=True)
        # self.log("F1_family_mfcc", f1_family_mfcc, prog_bar=True)
        # self.log("F1_instruments_mfcc", f1_instruments_mfcc, prog_bar=True)
        self.log("accuracy_combined", accuracy_combined, prog_bar=True)
        return {
            "loss": loss_family
            + loss_instruments
            + loss_family_mfcc
            + loss_instruments_mfcc,
        }

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def train_dataloader(self):
        return load_training_data(self.variant)

    def test_dataloader(self):
        return load_test_data(self.variant)

    def on_train_epoch_end(self):
        dict1 = {}
        dict2 = {}

        for key in self.training_step_target_family:
            dict1[key] = [
                self.training_step_preds_family[key],
                self.training_step_preds_family_mfcc[key],
                self.training_step_target_family[key],
            ]

        for key in self.training_step_target_instruments:
            dict2[key] = [
                self.training_step_preds_instruments[key],
                self.training_step_preds_instruments_mfcc[key],
                self.training_step_target_instruments[key],
            ]

        df1 = pd.DataFrame(dict1, index=["predicted_melspec", "predicted_mfcc", "true"])
        df2 = pd.DataFrame(dict2, index=["predicted_melspec", "predicted_mfcc", "true"])
        print(df1)
        print(df2)

        self.training_step_preds_family.clear()  # free memory
        self.training_step_target_instruments.clear()  # free memory
        self.training_step_preds_instruments.clear()  # free memory
        self.training_step_target_family.clear()  # free memory

    def score_level_fusion(
        self,
        prediction_family,
        prediction_instruments,
        prediction_family_mfcc,
        prediction_instruments_mfcc,
    ):
        # only melspec
        boosted_instruments = self.predict_instruments_p1(prediction_family)

        y_pred_2 = self.predict_instruments_p2(
            prediction_instruments, boosted_instruments
        )

        # only mfcc
        boosted_instruments_mfcc = self.predict_instruments_p1(prediction_family_mfcc)

        y_pred_2_mfcc = self.predict_instruments_p2(
            prediction_instruments_mfcc, boosted_instruments_mfcc
        )

        # combined
        boosted_instruments_combined = self.combine(
            mfcc=boosted_instruments_mfcc, melspec=boosted_instruments
        )

        prediction_instruments_combined = self.combine(
            melspec=prediction_instruments, mfcc=prediction_family_mfcc
        )

        y_pred_2_combined = self.predict_instruments_p2(
            prediction_instruments_combined, boosted_instruments_combined
        )

        return y_pred_2, y_pred_2_mfcc, y_pred_2_combined

    def predict_instruments_p1(self, prediction_family):
        y_pred_1 = []

        prediction_1 = prediction_family
        for sample in prediction_1:
            y_pred_1.append([1 if i >= 1e-01 else 0 for i in sample])

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

        return boosted_instruments  # most probable instruments based on family

    def predict_instruments_p2(self, prediction_instruments, boosted_instruments):
        y_pred_2 = []

        prediction_2 = prediction_instruments

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

        return y_pred_2  # predicted instruments compared with previous prediction

    def combine(self, melspec, mfcc):
        result = []
        for index in range(len(melspec)):
            print(type(melspec[index]))
            print(type(mfcc[index]))

            # melspec = np.concatenate(np.array(melspec[index]).flatten())
            # mfcc = np.concatenate(np.array(mfcc[index]).flatten())
            # print(melspec)
            # print(mfcc)
            # combined_array = np.concatenate(melspec, mfcc)
            # unique_values = np.unique(combined_array)
            # result.append(unique_values)

        print(result)
        return result
