from module import Module
from cnn import DefaultCNN as CNN
from lstm import DefaultLSTM as LSTM
from labels import prepare_labels, prepare_labels_short

import pytorch_lightning as pl


def training(case):
    true_labels_instruments = prepare_labels_short(1)
    true_labels_family = prepare_labels_short(0)

    learner = Module(
        model_family=CNN(4),
        model_instruments=CNN(11),
        # model_family=LSTM(num_classes=4, input_size=11, hidden_size=128),
        # model_instruments=LSTM(num_classes=11, input_size=11, hidden_size=128),
        true_labels_instruments=true_labels_instruments,
        true_labels_family=true_labels_family,
        variant=case,
    )

    # logger = TensorBoardLogger("logs/", name="logger")

    trainer.fit(learner)


def test(trainer):
    result = trainer.test(ckpt_path="best")
    print(result)


checkpoint = pl.callbacks.ModelCheckpoint(
    dirpath="./checkpoints", every_n_epochs=1, save_top_k=-1
)
trainer = pl.Trainer(
    accelerator="gpu",
    devices=1,
    max_epochs=10,
    callbacks=[checkpoint],
    check_val_every_n_epoch=1,
    # logger=logger,
    # log_every_n_steps=50,
)


# [ 1000 = 1 [sec] | 100 = 100 [ms] | 10 = 10 [ms] ]
# training(1000)
# training(100)
training(100)
test(trainer)
