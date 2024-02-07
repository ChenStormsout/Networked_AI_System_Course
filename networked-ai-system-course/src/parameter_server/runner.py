import json
from datetime import datetime
from typing import Dict, List

import numpy as np
from mqtt_builder import get_mqqt_client


class Runner:
    def __init__(self, meta_learning_mode: str) -> None:
        self.meta_learning_mode = meta_learning_mode
        self.mqtt_client = get_mqqt_client("server", self.respond)
        self.weights: None | np.ndarray = None
        self.hp_config = {
            "batch_size_mean": 64,
            "batch_size_std": 20,
            "learning_rate_mean": -3,
            "learning_rate_std": 1,
            "nesterov_mean": 0.5,
            "nesterov_std": 0.3,
            "momentum_mean": 0.5,
            "momentum_std": 0.3,
            "n_runs_mean": 10,
            "n_runs_std": 3,
            "n_epochs_mean": 10,
            "n_epochs_std": 3,
        }
        self.result_container: Dict = {}

    def respond(self, client, userdata, msg) -> None:
        """Method to override the on_message method of the Mqtt client.
        Defines the main behaviour of the server in regards to send messages."""
        payload = json.loads(msg.payload)
        # print("Got the payload: ", payload)
        if "target_topic" in payload:
            print("Target topic recieved:", payload["target_topic"])
            self.mqtt_client.publish(
                payload["target_topic"],
                json.dumps(dict(weights=self.weights, hp_config=self.hp_config)),
            )
        else:
            print("Trying to aggregate...")
            self.update_global_model(payload)

    def update_global_model(self, payload: Dict):
        """Method is work in progress...
        Has the tasks to
         - store the new weights and hp_configs in self.result_container
         - Clearing the result_container if more than 10 updates per
         node are existing
         - Creating a new global model based on the stored weights in
         self.result_container
        It may be reasonable to split it in submethods for better readabilty.

        Parameters
        ----------
        payload : Dict
            Dictionary containing the send data from the Mqtt message with
            a new training result.
        """
        id = payload["id"]
        self.maintain_result_container(id, payload)
        model_weights = []
        for node_id, node_results in self.result_container.items():
            model_weights += node_results["weights"]
        global_model = []
        for layer_index in range(len(model_weights[0])):
            layer_weights = np.array(
                [
                    single_model_weights[layer_index]
                    for single_model_weights in model_weights
                ]
            )
            # print(layer_weights.shape)
            global_model.append(np.mean(layer_weights, axis=0).tolist())
        self.weights = global_model  # payload["best_model_weights"] # Temporary test
        self.update_hp_config()

    def maintain_result_container(self, id, payload) -> None:
        if not id in self.result_container:
            self.result_container[id] = dict(
                timestamps=[datetime.now()],
                scores=[payload["best_score"]],
                weights=[payload["best_model_weights"]],
                hp_params=[payload["best_hyper_params"]],
            )
        else:
            self.result_container[id]["timestamps"].append(datetime.now())
            self.result_container[id]["scores"].append(payload["best_score"])
            self.result_container[id]["weights"].append(payload["best_model_weights"])
            self.result_container[id]["hp_params"].append(payload["best_hyper_params"])
        for node_id, node_results in self.result_container.items():
            n_datapoints = len(node_results["timestamps"])
            if n_datapoints > 5:
                print("Cutting of old data")
                for key, value in node_results.items():
                    node_results[key] = value[-5:]
            self.result_container[node_id] = node_results

    def update_hp_config(self) -> None:
        """Method to update the hp_config that is given the nodes as
        baseline to generate its hyper_parameters from.
        """
        hp_names = set(["_".join(key.split("_")[:-1]) for key in self.hp_config.keys()])
        hp_values = dict()
        scores = []
        for name in hp_names:
            hp_values[name] = []
        update_counter = 0
        for node_id, node_results in self.result_container.items():
            for name in hp_names:
                if name == "learning_rate":
                    hp_values[name] += [
                        np.log10(value[name]) for value in node_results["hp_params"]
                    ]
                else:
                    hp_values[name] += [
                        value[name] for value in node_results["hp_params"]
                    ]
            update_counter += len(node_results["hp_params"])
            scores += node_results["scores"]
        if update_counter >= 5:
            if self.meta_learning_mode == "DIRECT_MEANS":
                new_config = self.gradual_means_hp_update(
                    hp_values, hp_names, update_rate=1.0
                )
            elif self.meta_learning_mode == "DIRECT_SCORE_WEIGHTED_MEANS":
                new_config = self.gradual_score_weighted_means_hp_update(
                    hp_values, hp_names, scores, update_rate=1.0
                )
            elif self.meta_learning_mode == "GRADUAL_MEANS":
                new_config = self.gradual_means_hp_update(hp_values, hp_names, scores)
            elif self.meta_learning_mode == "GRADUAL_SCORE_WEIGHTED_MEANS":
                new_config = self.gradual_score_weighted_means_hp_update(
                    hp_values, hp_names, scores
                )
            elif self.meta_learning_mode == "DIRECT_GENETIC_ALGORITHM_MATING":
                new_config = self.genetic_algorithm_mating_hp_update(
                    hp_values, hp_names, scores,update_rate=1.
                )
            elif self.meta_learning_mode == "GRADUAL_GENETIC_ALGORITHM_MATING":
                new_config = self.genetic_algorithm_mating_hp_update(
                    hp_values, hp_names, scores
                )
            else:
                raise ValueError(
                    f"Invalid value for meta_learning_mode: \
                        {self.meta_learning_mode} was given."
                )
            print("New config:", new_config)
            self.hp_config = new_config

    def gradual_means_hp_update(
        self,
        hp_values: Dict[str, List[float]],
        hp_names: List[str],
        update_rate: float = 0.1,
    ) -> Dict[str, float]:
        """Recalculates the hp_config by directly calculating the average
        and standard deviation of the given training results in hp_values

        Parameters
        ----------
        hp_values : Dict[str, List[float]]
            Dictionary of used hyper-parameter values in the best training
            from the nodes. Key is the name of the hyper-parameter and
            values are a list containing the valuesused in the trainings.
        hp_names : List[str]
            Names of hyper-parameter that shall be updated.
        update_rate : float, optional
            Update rate for the hp_config update. Similar than learning rate in SGD,
              by default 0.1

        Returns
        -------
        Dict[str, float]
            New hp_config dictionary.
        """
        new_config = dict()
        for name in hp_names:
            new_config[name + "_mean"] = (1 - update_rate) * self.hp_config[
                name + "_mean"
            ] + update_rate * np.average(hp_values[name])
            new_config[name + "_std"] = (1 - update_rate) * self.hp_config[
                name + "_std"
            ] + update_rate * np.std(hp_values[name])
        return new_config

    def gradual_score_weighted_means_hp_update(
        self,
        hp_values: Dict[str, List[float]],
        hp_names: List[str],
        scores: List[float],
        update_rate: float = 0.1,
    ) -> Dict[str, float]:
        """Recalculates the hp_config by directly calculating the average
        and standard deviation of the given training results in hp_values.
        Thereby the average and standard deviation is weighted by the scores.

        Parameters
        ----------
        hp_values : Dict[str, List[float]]
            Dictionary of used hyper-parameter values in the best training
            from the nodes. Key is the name of the hyper-parameter and
            values are a list containing the valuesused in the trainings.
        hp_names : List[str]
            Names of hyper-parameter that shall be updated.
        scores : List[float]
            List of the training scores for the trained models.
        update_rate : float, optional
            Update rate for the hp_config update. Similar than learning rate in SGD,
            by default 0.1

        Returns
        -------
        Dict[str, float]
            New hp_config dictionary.
        """
        new_config = dict()
        for name in hp_names:
            new_config[name + "_mean"] = (1 - update_rate) * self.hp_config[
                name + "_mean"
            ] + update_rate * np.average(hp_values[name], weights=scores)
            # https://stackoverflow.com/questions/2413522/weighted-standard-deviation-in-numpy
            new_config[name + "_std"] = (1 - update_rate) * self.hp_config[
                name + "_std"
            ] + update_rate * np.sqrt(
                np.cov(scores, aweights=scores)
            )  # np.std(hp_values[name])
        return new_config

    def genetic_algorithm_mating_hp_update(
        self,
        hp_values: Dict[str, List[float]],
        hp_names: List[str],
        scores: List[float],
        update_rate: float = 0.1,
    ) -> Dict[str, float]:
        """Meta learning that first draws a population of the hp values
        of 500 entities weighted by the scores.

        Parameters
        ----------
        hp_values : Dict[str, List[float]]
            Dictionary of used hyper-parameter values in the best training
            from the nodes. Key is the name of the hyper-parameter and
            values are a list containing the valuesused in the trainings.
        hp_names : List[str]
            Names of hyper-parameter that shall be updated.
        scores : List[float]
            List of the training scores for the trained models.
        update_rate : float, optional
            Update rate for the hp_config update. Similar than learning rate in SGD,
            by default 0.1

        Returns
        -------
        Dict[str, float]
            New hp_config dictionary.
        """
        new_config = dict()
        for name in hp_names:
            population = np.random.choice(hp_values[name], size=500, p=np.array(scores)/sum(scores))
            new_config[name + "_mean"] = (1 - update_rate) * self.hp_config[
                name + "_mean"
            ] + update_rate * np.average(population)
            new_config[name + "_std"] = (1 - update_rate) * self.hp_config[
                name + "_std"
            ] + update_rate * np.std(population)
        return new_config

    def run(self) -> None:
        """Main method causing the runner to have its behaviour.
        For this runner, the behaviour is listening to mqtt messages and
        react on them. Therefore, run() only calls the loop_forever() method
        of the mqtt client.
        """
        self.mqtt_client.loop_forever()
