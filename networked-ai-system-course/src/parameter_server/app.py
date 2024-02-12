import os
import numpy as np
import tensorflow as tf

RANDOM_SEED = int(os.environ["RANDOM_SEED"])
np.random.seed(RANDOM_SEED)
tf.keras.utils.set_random_seed(RANDOM_SEED)

from runner import Runner

# Choose from META_LEARNING_MODE from the following:
# - DIRECT_MEANS
# - DIRECT_SCORE_WEIGHTED_MEANS
# - DIRECT_GENETIC_ALGORITHM_MATING
# - GRADUAL_MEANS
# - GRADUAL_SCORE_WEIGHTED_MEANS
# - GRADUAL_GENETIC_ALGORITHM_MATING

META_LEARNING_MODE = os.getenv("meta_learning_mode")
if META_LEARNING_MODE is None:
    raise ValueError("No meta learning mode specified.")


# Choose from AGGREGATION_METHOD from the following:
AGGREGATION_METHOD = "WEIGHTED AVERAGE - RECENCY"
#WEIGHTED AVERAGE - RECENCY
#WEIGHTED AVERAGE - PERFORMANCE SCORES

r = Runner(META_LEARNING_MODE, AGGREGATION_METHOD)
r.run()