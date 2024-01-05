from module import Module
from cnn import SimpleCNN, DefaultCNN
from rnn import DefaultRNN, SimpleRNN
from lstm import DefaultLSTM
from labels import prepare_labels, prepare_labels_short

import pytorch_lightning as pl

NUM_EPOCHS = 20


def training(case, f1, f2):
    true_labels_instruments = prepare_labels_short(1)
    true_labels_family = prepare_labels_short(0)

    # RNN + LSTM
    if case == 1000:
        input_size = 101
    if case == 100:
        input_size = 11
    if case == 10:
        input_size = 2

    if (f1 or f2) == (SimpleCNN or DefaultCNN):
        model_family = f1(4, case)
        model_instruments = f2(11, case)
    else:
        model_family = (f1(num_classes=4, input_size=input_size, hidden_size=128),)
        model_instruments = (
            f2(num_classes=11, input_size=input_size, hidden_size=128),
        )

    learner = Module(
        model_family=model_family,
        model_instruments=model_instruments,
        true_labels_instruments=true_labels_instruments,
        true_labels_family=true_labels_family,
        variant=case,
        num_epochs=NUM_EPOCHS,
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
    max_epochs=NUM_EPOCHS,
    callbacks=[checkpoint],
    check_val_every_n_epoch=1,
    # logger=logger,
    # log_every_n_steps=50,
)


# [ 1000 = 1 [sec] | 100 = 100 [ms] | 10 = 10 [ms] ]

training(1000, SimpleCNN, SimpleCNN)
test(trainer)

# training(1000, SimpleCNN, DefaultCNN)
# test(trainer)

# training(1000, SimpleCNN, SimpleRNN)
# test(trainer)

# training(1000, SimpleCNN, DefaultRNN)
# test(trainer)

# training(1000, SimpleCNN, DefaultLSTM)
# test(trainer)

# # ------

# training(1000, DefaultCNN, SimpleCNN)
# test(trainer)

# training(1000, DefaultCNN, DefaultCNN)
# test(trainer)

# training(1000, DefaultCNN, SimpleRNN)
# test(trainer)

# training(1000, DefaultCNN, DefaultRNN)
# test(trainer)

# training(1000, DefaultCNN, DefaultLSTM)
# test(trainer)

# # ------

# training(1000, SimpleRNN, SimpleCNN)
# test(trainer)

# training(1000, SimpleRNN, DefaultCNN)
# test(trainer)

# training(1000, SimpleRNN, SimpleRNN)
# test(trainer)

# training(1000, SimpleRNN, DefaultRNN)
# test(trainer)

# training(1000, SimpleRNN, DefaultLSTM)
# test(trainer)

# # ------

# training(1000, DefaultRNN, SimpleCNN)
# test(trainer)

# training(1000, DefaultRNN, DefaultCNN)
# test(trainer)

# training(1000, DefaultRNN, SimpleRNN)
# test(trainer)

# training(1000, DefaultRNN, DefaultRNN)
# test(trainer)

# training(1000, DefaultRNN, DefaultLSTM)
# test(trainer)

# # ------

# training(1000, DefaultLSTM, SimpleCNN)
# test(trainer)

# training(1000, DefaultLSTM, DefaultCNN)
# test(trainer)

# training(1000, DefaultLSTM, SimpleRNN)
# test(trainer)

# training(1000, DefaultLSTM, DefaultRNN)
# test(trainer)

# training(1000, DefaultLSTM, DefaultLSTM)
# test(trainer)
