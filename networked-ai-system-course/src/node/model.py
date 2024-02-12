import os
import platform, sys
from typing import List
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # surpressing tensorflow spam messages

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input, LeakyReLU
from tensorflow.keras.losses import BinaryCrossentropy

if platform.processor() == "arm" and sys.platform == "darwin":
    from tensorflow.keras.optimizers.legacy import SGD
else:
    from tensorflow.keras.optimizers.experimental import SGD


from loguru import logger


logger.add(sys.stderr, format="{time} - {level} - {message}", level="DEBUG")


def get_model(
    weights: None | List = None,
    learning_rate: float = 0.01,
    momentum: float = 0.9,
    nesterov: bool = True,
) -> Sequential:
    """Generates the basic model used in each node.
    Consists of an Input with shape 2, three dense layers
    with 4 units each and LeakyRelu(alpha=0.1) as activation function,
    and final dense layer with one unit and sigmoid acitvation function.
    As optimizer, SGD is used. As loss, binary cross entropy.

    Parameters
    ----------
    weights : None | List, optional
        Weights that shall be set in the model. If none, random
        weights are used, by default None
    learning_rate : float, optional
        Learning rate of the SGD optimizer, by default 0.01
    momentum : float, optional
        Momentum of the SGD optimizer, by default 0.9
    nesterov : bool, optional
        Whether to calculate nesterov version of momentum or not,
        by default True

    Returns
    -------
    Sequential
        tf.keras.Sequential model ready to be trained and
        set hyperparameters.
    """
    model = Sequential()
    model.add(Input(shape=(2,)))
    model.add(Dense(units=4, activation=LeakyReLU(alpha=0.1)))
    model.add(Dense(units=4, activation=LeakyReLU(alpha=0.1)))
    model.add(Dense(units=4, activation=LeakyReLU(alpha=0.1)))
    model.add(Dense(units=1, activation="sigmoid"))
    model.compile(
        optimizer=SGD(learning_rate=0.01, momentum=0.9, nesterov=True),
        loss=BinaryCrossentropy(),
        metrics=[
            "accuracy",
        ],
    )
    if not weights is None:
        model.set_weights(weights)
    return model


if __name__ == "__main__":
    model = get_model()
    logger.info(f"Model summary: {model.summary()}")
