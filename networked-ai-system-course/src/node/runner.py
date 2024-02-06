import os
import time
from typing import Dict, List, Tuple

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # surpressing tensorflow spam messages
import json
from uuid import uuid1

import numpy as np
from dataset_generator import DatasetGenerator
from model import get_model
from mqtt_builder import get_mqqt_client
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential


class Runner:
    def __init__(self) -> None:
        """This is the main class that runs in each node.
        It contains everything that is required to execute the meta-learning
        federated learning including communication from the node's perspective.
        """
        self.dg: DatasetGenerator = DatasetGenerator()
        self.latest_test_score: float = 1
        self.model: None | Sequential = None
        self.retrieved_update: bool = False
        self.weights: np.ndarray[float] | None = None
        self.hp_config: Dict = {
            "batch_size_mean": 64,
            "batch_size_std": 20,
            "learning_rate_mean": -3,
            "learning_rate_std": 1,
            "nesterov_mean": 0.5,
            "nesterov_std": 0.3,
            "momentum_mean": 0.5,
            "momentum_std": 0.3,
        }
        self.id = str(uuid1())
        self.mqtt_client = get_mqqt_client(self.id, self.retrieve_global_model)

    def train_models(
        self, X: np.ndarray[float], y: np.ndarray[float]
    ) -> Dict[str, float | List[float] | List[np.ndarray]]:
        """Executes n_runs trainings with random hyperparameters each
        runs and returns the best score, the best hyperparameters and
        the model weights.

        Returns
        -------
        Dict[str, float | List[float]| List[np.ndarray]]
            The best score, the best hyperparameters and
            the model weights
        """
        self.get_global_model()
        n_runs = 5
        n_epochs = 5
        models = []
        hyper_params = []
        scores = []
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        for _ in range(n_runs):
            hyper_param_dict = self.suggest_hyper_params(self.hp_config)
            model = get_model(
                self.weights,
                hyper_param_dict["learning_rate"],
                hyper_param_dict["momentum"],
                hyper_param_dict["nesterov"],
            )
            if _ == 0:
                print(
                    "Pre-training performance: ",
                    accuracy_score(y_test, model.predict(X_test, verbose=0) > 0.5),
                )
            model.fit(
                X_train,
                y_train,
                batch_size=hyper_param_dict["batch_size"],
                epochs=n_epochs,
                verbose=0,
            )
            models.append(model)
            scores.append(
                accuracy_score(y_test, model.predict(X_test, verbose=0) > 0.5)
            )
            hyper_params.append(hyper_param_dict)
        best_model_idx = np.argmax(scores)

        results = dict()
        results["best_score"] = scores[best_model_idx]
        results["best_hyper_params"] = hyper_params[best_model_idx]
        results["best_model_weights"] = [
            arr.tolist() for arr in models[best_model_idx].get_weights()
        ]
        # print(models[best_model_idx].get_weights())
        self.model = models[best_model_idx]
        self.latest_test_score = results["best_score"]
        print("Best_score: ", results["best_score"])
        return results

    def suggest_hyper_params(self, config: Dict) -> Dict:
        """Suggests the following hyperparams based on a
        config giving mean and std of a gaussian distribution,
        from which the hyperparams are then randomized.
        Gives the following hyperparams:
         - learning_rate
         - momentum
         - nesterov
         - batch_size
        For each hyperparam, the config dict must contain a mean
        and std value. Naming convention is <hyperparam name>_mean and
        <hyperparam name>_std, e.g. learning_rate_mean and batch_size_mean.

        Parameters
        ----------
        config : Dict
            Config dict defining with what params the hyperparams are supposed to
            be randomed. For each hyperparam, the config dict must contain a mean
            and std value. Naming convention is <hyperparam name>_mean and
            <hyperparam name>_std, e.g. learning_rate_mean and batch_size_mean.

        Returns
        -------
        Dict
            Returns a dict with the hyperparameter as key and the suggested value
            as the value.
        """
        suggested_params = dict()
        suggested_params["learning_rate"] = 10 ** np.clip(
            np.random.normal(
                loc=config["learning_rate_mean"], scale=config["learning_rate_std"]
            ),
            a_min=-5,
            a_max=1,
        )
        suggested_params["batch_size"] = int(
            np.clip(
                np.random.normal(
                    loc=config["batch_size_mean"], scale=config["batch_size_std"]
                ),
                a_min=2,
                a_max=512,
            )
        )
        suggested_params["nesterov"] = bool(
            np.clip(
                np.random.normal(
                    loc=config["nesterov_mean"], scale=config["nesterov_std"]
                ),
                a_min=0,
                a_max=1,
            )
            > 0.5
        )
        suggested_params["momentum"] = np.clip(
            np.random.normal(loc=config["momentum_mean"], scale=config["momentum_std"]),
            a_min=0,
            a_max=1,
        )
        return suggested_params

    def run(self) -> None:
        """Blocking method that executes the node behaviour.
        The behaviour consists of the following steps:
         - Generate new data
         - Train a new model, if there is none yet, or if the performance
           dropped by 10% compared to the latest performance on the test
           set
         - If a model is trained, communicate the results to the central server
         - Sleep 0.5 seconds"""
        iter_counter = 0
        print("Starting to run.")
        while True:
            iter_counter += 1
            X, y = self.dg()
            do_trainig = False
            if self.model is None:
                do_trainig = True
            else:
                curr_score = accuracy_score(y, self.model.predict(X, verbose=0) > 0.5)
                if curr_score < self.latest_test_score * 0.9:
                    do_trainig = True
            if do_trainig:
                print(f"Training a new model at iteration {iter_counter}")
                training_result = self.train_models(X, y)
                self.communicate_update(training_result)  # tbd
            time.sleep(0.5)

    def get_global_model(self) -> None:
        """Method to simulate pull behaviour in mqtt, which is
        actually only push.
        In this method, we send a message to the server with our subscripted
        topic. This shall trigger the server to publish a message with the
        most recent model weights and hyper_parameter configs. This method
        waits maximum one second for the answer of the server.
        """
        self.retrieved_update = False
        self.mqtt_client.publish(
            "server_get_model", json.dumps(dict(target_topic=self.id))
        )
        for _ in range(10):
            if self.retrieved_update:
                break
            time.sleep(0.1)
        if not self.retrieved_update:
            print("Warning: Did not retrieve global model in time.")

    def retrieve_global_model(self, client, userdata, msg) -> None:
        """Method that is used to override the on_message of the
        mqtt client. It expects weights and hp_config in the message and
        overrides the current values of the runner. Also self.retrieved_update is
        set to True to indacte we got an update."""
        payload = json.loads(msg.payload)
        if not payload["weights"] is None:
            self.weights = [
                np.array(layer_weights) for layer_weights in payload["weights"]
            ]
        else:
            self.weights = None
        self.hp_config = payload["hp_config"]
        self.retrieved_update = True

    def communicate_update(self, training_results: Tuple) -> None:
        """Sends training results to the server.

        Parameters
        ----------
        training_results : Tuple
            Training results returned by self.train_models.
        """
        print("Sending training results to server")
        training_results["id"] = self.id
        self.mqtt_client.publish("server_update", json.dumps(training_results))


if __name__ == "__main__":
    r = Runner()
    r.run()
