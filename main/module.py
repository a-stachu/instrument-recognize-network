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
from collections import Counter
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
        num_epochs,
        learning_rate=1e-3,
    ):
        super().__init__()

        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

        self.model_instruments = model_instruments
        self.model_family = model_family
        self.true_labels_instruments = true_labels_instruments
        self.true_labels_family = true_labels_family

        self.variant = variant

        self.epoch = 1  # current epoch
        self.temporary = []
        self.dict1 = {}
        self.dict2 = {}

        self.training_step_preds_instruments = {}
        self.training_step_target_instruments = {}
        self.training_step_preds_family = {}
        self.training_step_target_family = {}

        self.training_step_preds_instruments_mfcc = {}
        self.training_step_target_instruments_mfcc = {}
        self.training_step_preds_family_mfcc = {}
        self.training_step_target_family_mfcc = {}

        # ------------------------ [TRENING]
        self.arr_train_loss_family = []
        self.arr_train_loss_instruments = []
        self.arr_train_loss_family_mfcc = []
        self.arr_train_loss_instruments_mfcc = []

        self.arr_train_precision_family = []
        self.arr_train_precision_instruments = []
        self.arr_train_precision_family_mfcc = []
        self.arr_train_precision_instruments_mfcc = []

        self.arr_train_recall_family = []
        self.arr_train_recall_instruments = []
        self.arr_train_recall_family_mfcc = []
        self.arr_train_recall_instruments_mfcc = []

        self.arr_train_f1_family = []
        self.arr_train_f1_instruments = []
        self.arr_train_f1_family_mfcc = []
        self.arr_train_f1_instruments_mfcc = []

        self.arr_train_accuracy_family = []
        self.arr_train_accuracy_instruments = []
        self.arr_train_accuracy_family_mfcc = []
        self.arr_train_accuracy_instruments_mfcc = []
        self.arr_train_accuracy = []
        self.arr_train_accuracy_mfcc = []
        self.arr_train_accuracy_combined = []

        # ------------------------ [TEST]
        self.arr_test_loss_family = []
        self.arr_test_loss_instruments = []
        self.arr_test_loss_family_mfcc = []
        self.arr_test_loss_instruments_mfcc = []

        self.arr_test_precision_family = []
        self.arr_test_precision_instruments = []
        self.arr_test_precision_family_mfcc = []
        self.arr_test_precision_instruments_mfcc = []

        self.arr_test_recall_family = []
        self.arr_test_recall_instruments = []
        self.arr_test_recall_family_mfcc = []
        self.arr_test_recall_instruments_mfcc = []

        self.arr_test_f1_family = []
        self.arr_test_f1_instruments = []
        self.arr_test_f1_family_mfcc = []
        self.arr_test_f1_instruments_mfcc = []

        self.arr_test_accuracy_family = []
        self.arr_test_accuracy_instruments = []
        self.arr_test_accuracy_family_mfcc = []
        self.arr_test_accuracy_instruments_mfcc = []
        self.arr_test_accuracy = []
        self.arr_test_accuracy_mfcc = []
        self.arr_test_accuracy_combined = []

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

        if self.epoch == 1:
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
            threshold=1e-01,
        )
        populate_table(
            tensor_true1,
            tensor_predicted1_mfcc,
            self.training_step_target_family_mfcc,
            instruments_group_short,
            threshold=0,
        )

        populate_table(
            tensor_true2,
            tensor_predicted2_mfcc,
            self.training_step_preds_instruments_mfcc,
            instruments,
            expanded=instruments_short,
            threshold=1e-01,
        )
        populate_table(
            tensor_true2,
            tensor_predicted2_mfcc,
            self.training_step_target_instruments_mfcc,
            instruments,
            expanded=instruments_short,
            threshold=0,
        )

        populate_table(
            tensor_true1,
            tensor_predicted1,
            self.training_step_preds_family,
            instruments_group_short,
            threshold=1e-01,
        )
        populate_table(
            tensor_true1,
            tensor_predicted1,
            self.training_step_target_family,
            instruments_group_short,
            threshold=0,
        )

        populate_table(
            tensor_true2,
            tensor_predicted2,
            self.training_step_preds_instruments,
            instruments,
            expanded=instruments_short,
            threshold=1e-01,
        )
        populate_table(
            tensor_true2,
            tensor_predicted2,
            self.training_step_target_instruments,
            instruments,
            expanded=instruments_short,
            threshold=0,
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

        self.temporary = [
            loss_family.item(),
            loss_instruments.item(),
            loss_family_mfcc.item(),
            loss_instruments_mfcc.item(),
            precision_family,
            precision_instruments,
            precision_family_mfcc,
            precision_instruments_mfcc,
            recall_family,
            recall_instruments,
            recall_family_mfcc,
            recall_instruments_mfcc,
            f1_family,
            f1_instruments,
            f1_family_mfcc,
            f1_instruments_mfcc,
            accuracy_family.item(),
            accuracy_instrument.item(),
            accuracy_family_mfcc.item(),
            accuracy_instrument_mfcc.item(),
            accuracy.item(),
            accuracy_mfcc.item(),
            accuracy_combined.item(),
        ]

        self.log("accuracy", accuracy, prog_bar=True)
        self.log("accuracy_instrument", accuracy_instrument, prog_bar=True)
        self.log("accuracy_family", accuracy_family, prog_bar=True)
        self.log("precision_family", precision_family, prog_bar=True)
        self.log("precision_instruments", precision_instruments, prog_bar=True)
        self.log("recall_family", recall_family, prog_bar=True)
        self.log("recall_instruments", recall_instruments, prog_bar=True)
        self.log("F1_family", f1_family, prog_bar=True)
        self.log("F1_instruments", f1_instruments, prog_bar=True)

        self.log("accuracy_mfcc", accuracy_mfcc, prog_bar=True)
        self.log("accuracy_instrument_mfcc", accuracy_instrument_mfcc, prog_bar=True)
        self.log("accuracy_family_mfcc", accuracy_family_mfcc, prog_bar=True)
        self.log("precision_family_mfcc", precision_family_mfcc, prog_bar=True)
        self.log(
            "precision_instruments_mfcc", precision_instruments_mfcc, prog_bar=True
        )
        self.log("recall_family_mfcc", recall_family_mfcc, prog_bar=True)
        self.log("recall_instruments_mfcc", recall_instruments_mfcc, prog_bar=True)
        self.log("F1_family_mfcc", f1_family_mfcc, prog_bar=True)
        self.log("F1_instruments_mfcc", f1_instruments_mfcc, prog_bar=True)
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

        print(accuracy_combined.cpu().item())

        df4 = pd.DataFrame(
            {
                "test_loss_family": loss_family.item(),
                "test_loss_instruments": loss_instruments.item(),
                "test_loss_family_mfcc": loss_family_mfcc.item(),
                "test_loss_instruments_mfcc": loss_instruments_mfcc.item(),
                "test_precision_family": precision_family,
                "test_precision_instruments": precision_instruments,
                "test_precision_family_mfcc": precision_family_mfcc,
                "test_precision_instruments_mfcc": precision_instruments_mfcc,
                "test_recall_family": recall_family,
                "test_recall_instruments": recall_instruments,
                "test_recall_family_mfcc": recall_family_mfcc,
                "test_recall_instruments_mfcc": recall_instruments_mfcc,
                "test_f1_family": f1_family,
                "test_f1_instruments": f1_instruments,
                "test_f1_family_mfcc": f1_family_mfcc,
                "test_f1_instruments_mfcc": f1_instruments_mfcc,
                "test_accuracy_family": accuracy_family.item(),
                "test_accuracy_instruments": accuracy_instrument.item(),
                "test_accuracy_family_mfcc": accuracy_family_mfcc.item(),
                "test_accuracy_instruments_mfcc": accuracy_instrument_mfcc.item(),
                "test_accuracy": accuracy.item(),
                "test_accuracy_mfcc": accuracy_mfcc.item(),
                "test_accuracy_combined": accuracy_combined.item(),
            },
            index=["value"],
        )

        df4.to_csv(
            f"TEST_{self.variant}_{self.num_epochs}_family_{type(self.model_family).__name__}_instruments_{type(self.model_instruments).__name__}.csv"
        )

        self.log("accuracy", accuracy, prog_bar=True)
        self.log("accuracy_instrument", accuracy_instrument, prog_bar=True)
        self.log("accuracy_family", accuracy_family, prog_bar=True)
        self.log("precision_family", precision_family, prog_bar=True)
        self.log("precision_instruments", precision_instruments, prog_bar=True)
        self.log("recall_family", recall_family, prog_bar=True)
        self.log("recall_instruments", recall_instruments, prog_bar=True)
        self.log("F1_family", f1_family, prog_bar=True)
        self.log("F1_instruments", f1_instruments, prog_bar=True)

        self.log("accuracy_mfcc", accuracy_mfcc, prog_bar=True)
        self.log("accuracy_instrument_mfcc", accuracy_instrument_mfcc, prog_bar=True)
        self.log("accuracy_family_mfcc", accuracy_family_mfcc, prog_bar=True)
        self.log("precision_family_mfcc", precision_family_mfcc, prog_bar=True)
        self.log(
            "precision_instruments_mfcc", precision_instruments_mfcc, prog_bar=True
        )
        self.log("recall_family_mfcc", recall_family_mfcc, prog_bar=True)
        self.log("recall_instruments_mfcc", recall_instruments_mfcc, prog_bar=True)
        self.log("F1_family_mfcc", f1_family_mfcc, prog_bar=True)
        self.log("F1_instruments_mfcc", f1_instruments_mfcc, prog_bar=True)
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
        for key in self.training_step_target_family:
            self.dict1[key] = [
                self.training_step_preds_family[key],
                self.training_step_preds_family_mfcc[key],
                self.training_step_target_family[key],
            ]

        for key in self.training_step_target_instruments:
            self.dict2[key] = [
                self.training_step_preds_instruments[key],
                self.training_step_preds_instruments_mfcc[key],
                self.training_step_target_instruments[key],
            ]

        # ------------------------ [TRENING]
        self.arr_train_loss_family.append(self.temporary[0])
        self.arr_train_loss_instruments.append(self.temporary[1])
        self.arr_train_loss_family_mfcc.append(self.temporary[2])
        self.arr_train_loss_instruments_mfcc.append(self.temporary[3])

        self.arr_train_precision_family.append(self.temporary[4])
        self.arr_train_precision_instruments.append(self.temporary[5])
        self.arr_train_precision_family_mfcc.append(self.temporary[6])
        self.arr_train_precision_instruments_mfcc.append(self.temporary[7])

        self.arr_train_recall_family.append(self.temporary[8])
        self.arr_train_recall_instruments.append(self.temporary[9])
        self.arr_train_recall_family_mfcc.append(self.temporary[10])
        self.arr_train_recall_instruments_mfcc.append(self.temporary[11])

        self.arr_train_f1_family.append(self.temporary[12])
        self.arr_train_f1_instruments.append(self.temporary[13])
        self.arr_train_f1_family_mfcc.append(self.temporary[14])
        self.arr_train_f1_instruments_mfcc.append(self.temporary[15])

        self.arr_train_accuracy_family.append(self.temporary[16])
        self.arr_train_accuracy_instruments.append(self.temporary[17])
        self.arr_train_accuracy_family_mfcc.append(self.temporary[18])
        self.arr_train_accuracy_instruments_mfcc.append(self.temporary[19])
        self.arr_train_accuracy.append(self.temporary[20])
        self.arr_train_accuracy_mfcc.append(self.temporary[21])
        self.arr_train_accuracy_combined.append(self.temporary[22])

        self.temporary.clear()

        # self.training_step_preds_family.clear()  # free memory
        # self.training_step_target_instruments.clear()  # free memory
        # self.training_step_preds_instruments.clear()  # free memory
        # self.training_step_target_family.clear()  # free memory

        # self.training_step_preds_instruments_mfcc.clear()
        # self.training_step_target_instruments_mfcc.clear()
        # self.training_step_preds_family_mfcc.clear()
        # self.training_step_target_family_mfcc.clear()

        self.epoch += 1

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
            mfcc=prediction_instruments_mfcc, melspec=prediction_instruments
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
                boosted_instruments.append(np.array([]))

        return boosted_instruments  # most probable instruments based on family

    def predict_instruments_p2(self, prediction_instruments, boosted_instruments):
        y_pred_2 = []

        prediction_2 = prediction_instruments

        # print(prediction_instruments, boosted_instruments)

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
            try:
                combined_array = np.concatenate(
                    (melspec[index].flatten(), mfcc[index].flatten())
                )
            except IndexError as e:
                if "index 2 is out of bounds for dimension 0 with size 1" in str(e):
                    empty_array = np.array([], dtype=np.float64)
                    if np.array_equal(melspec, empty_array):
                        combined_array = mfcc
                    else:
                        combined_array = melspec
                    result.append(np.unique(combined_array))
                else:
                    result.append(np.unique(combined_array))
            except TypeError as e:
                mean_tensor = melspec + mfcc / 2
                result = mean_tensor
            else:
                result.append(np.unique(combined_array))

        return result

    def on_train_end(self):
        df1 = pd.DataFrame(
            self.dict1, index=["predicted_melspec", "predicted_mfcc", "true"]
        )
        df2 = pd.DataFrame(
            self.dict2, index=["predicted_melspec", "predicted_mfcc", "true"]
        )
        df1.to_csv(
            f"TRAINING_{self.variant}_{self.num_epochs}_family_{type(self.model_family).__name__}_table.csv"
        )

        df2.to_csv(
            f"TRAINING_{self.variant}_{self.num_epochs}_instruments_{type(self.model_instruments).__name__}_table.csv"
        )

        df3 = pd.DataFrame(
            {
                "train_loss_family": self.arr_train_loss_family,
                "train_loss_instruments": self.arr_train_loss_instruments,
                "train_loss_family_mfcc": self.arr_train_loss_family_mfcc,
                "train_loss_instruments_mfcc": self.arr_train_loss_instruments_mfcc,
                "train_precision_family": self.arr_train_precision_family,
                "train_precision_instruments": self.arr_train_precision_instruments,
                "train_precision_family_mfcc": self.arr_train_precision_family_mfcc,
                "train_precision_instruments_mfcc": self.arr_train_precision_instruments_mfcc,
                "train_recall_family": self.arr_train_recall_family,
                "train_recall_instruments": self.arr_train_recall_instruments,
                "train_recall_family_mfcc": self.arr_train_recall_family_mfcc,
                "train_recall_instruments_mfcc": self.arr_train_recall_instruments_mfcc,
                "train_f1_family": self.arr_train_f1_family,
                "train_f1_instruments": self.arr_train_f1_instruments,
                "train_f1_family_mfcc": self.arr_train_f1_family_mfcc,
                "train_f1_instruments_mfcc": self.arr_train_f1_instruments_mfcc,
                "train_accuracy_family": self.arr_train_accuracy_family,
                "train_accuracy_instruments": self.arr_train_accuracy_instruments,
                "train_accuracy_family_mfcc": self.arr_train_accuracy_family_mfcc,
                "train_accuracy_instruments_mfcc": self.arr_train_accuracy_instruments_mfcc,
                "train_accuracy": self.arr_train_accuracy,
                "train_accuracy_mfcc": self.arr_train_accuracy_mfcc,
                "train_accuracy_combined": self.arr_train_accuracy_combined,
            }
        )

        index_labels = []
        for i in range(len(df3)):
            index_labels.append(i + 1)

        df3.index = index_labels
        df3.to_csv(
            f"TRAINING_{self.variant}_{self.num_epochs}_family_{type(self.model_family).__name__}_instruments_{type(self.model_instruments).__name__}.csv"
        )
