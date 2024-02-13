import os
import tensorflow as tf
import numpy as np

RANDOM_SEED = int(os.environ["RANDOM_SEED"])
np.random.seed(RANDOM_SEED)

tf.keras.utils.set_random_seed(RANDOM_SEED)

from runner import Runner

r = Runner()
r.run()
