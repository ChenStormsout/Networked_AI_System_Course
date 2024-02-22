import json
import os
import pickle
import sys
import time
from copy import deepcopy
from datetime import datetime, timedelta
from itertools import chain
from pathlib import Path
from typing import Dict, List

import numpy as np
from loguru import logger
from mqtt_builder import get_mqqt_client
logger.remove()
logger.add(sys.stderr, format="{time} - {level} - {message}", level="INFO")
# logging.basicConfig(
#     format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.DEBUG
# )
# logger = logging.getLogger(__name__)

LOG_PATH = os.getenv("log_path")
if LOG_PATH is None:
    LOG_PATH = Path("./networked-ai-system-course/tmp/bash")
else:
    LOG_PATH = Path(LOG_PATH)


class Runner:
    def __init__(self, meta_learning_mode: str, aggregation_method: str) -> None:
        self.meta_learning_mode = meta_learning_mode
        self.aggregation_method = aggregation_method
        self.mqtt_client = get_mqqt_client("server", self.respond)
        self.weights: None | List = None
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
        initial_params = deepcopy(self.hp_config)
        initial_params["weights"] = self.weights
        initial_params["meta_learning_mode"] = self.meta_learning_mode
        initial_params["timestamps"] = str(datetime.now())
        # os.mkdir(LOG_PATH / "parameter_server/")
        (LOG_PATH / "parameter_server").mkdir(parents=True, exist_ok=True)
        pickle.dump(
            initial_params,
            open(LOG_PATH / "parameter_server/initialization.pkl", mode="wb"),
        )

    def respond(self, client, userdata, msg) -> None:
        """Method to override the on_message method of the Mqtt client.
        Defines the main behaviour of the server in regards to send messages."""
        payload = json.loads(msg.payload)
        logger.info(f"Got the payload: {payload}")
        if "target_topic" in payload:
            logger.info(f'Target topic recieved: {payload["target_topic"]}')
            self.mqtt_client.publish(
                payload["target_topic"],
                json.dumps(dict(weights=self.weights, hp_config=self.hp_config)),
            )
        else:
            logger.info(f"Trying to aggregate with {self.aggregation_method}...")
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

        if(self.aggregation_method=="AVERAGE_AGGREGATION"):
            global_model=self.average_aggregation(model_weights)
        elif(self.aggregation_method=="MEDIAN_AGGREGATION"):
            global_model=self.median_aggregation(model_weights)
        elif(self.aggregation_method=="DYNAMIC_CLIPPED_AVERAGE_AGGREGATION"):
            global_model=self.dynamic_clip_aggregation(model_weights)
        elif(self.aggregation_method=="FIXED_CLIPPED_AVERAGE_AGGREGATION"):
            global_model=self.fixed_clip_aggregation(model_weights)
        elif(self.aggregation_method=="WEIGHTED_AVERAGE_PERFORMANCE_SCORES"):
            model_scores = []
            for node_results in self.result_container.values():
                model_scores += node_results["scores"]
            score_weights = np.exp(model_scores) / np.sum(np.exp(model_scores))
            global_model=self.weighted_score_aggregation(score_weights, model_weights)
        elif(self.aggregation_method=="WEIGHTED_AVERAGE_RECENCY"):
            decay_rate = 0.9
            timestamps = chain.from_iterable([node_results["timestamps"] for node_results in self.result_container.values()])
            time_diff_seconds = np.array([(datetime.now() - timestamp).total_seconds() for timestamp in timestamps])
            recency_weights = np.exp(-decay_rate * time_diff_seconds)
            global_model=self.weighted_recency_aggregation(recency_weights,model_weights)
        elif(self.aggregation_method=="WEIGHTED_AVERAGE_PUNISHING_UPDATES"):
            current_time = datetime.now()
            minutes_ago = current_time - timedelta(minutes=1)
            recent_update_counts = sum(1 for timestamp in chain.from_iterable([node_results["timestamps"] for node_results in self.result_container.values()])
                                   if timestamp >= minutes_ago)            
            punish_weights = 1 / (recent_update_counts + 1)
            global_model=self.weighted_punish_aggregation(punish_weights,model_weights)
        self.weights = global_model  # payload["best_model_weights"] # Temporary test
        self.update_hp_config()
        self.log_step()
            

    def average_aggregation(self, model_weights):
        result = []
        for layer_index in range(len(model_weights[0])):
            layer_weights = np.array(
                [
                    single_model_weights[layer_index]
                    for single_model_weights in model_weights
                ]
            )
            result.append(np.mean(layer_weights, axis=0).tolist())
        return result
        
    def median_aggregation(self, model_weights):
        result = []
        for layer_index in range(len(model_weights[0])):
            layer_weights = np.array(
                [
                    single_model_weights[layer_index]
                    for single_model_weights in model_weights
                ]
            )
            result.append(np.median(layer_weights, axis=0).tolist())
        return result
        
    def dynamic_clip_aggregation(self, model_weights):
        result = []
        for layer_index in range(len(model_weights[0])):
            layer_weights = np.array(
                [
                    single_model_weights[layer_index]
                    for single_model_weights in model_weights
                ]
            )
            lower_quantile, upper_quantile = np.percentile(layer_weights, [25, 75], axis=0)
            delta = upper_quantile - lower_quantile
            lower_clip_value = lower_quantile - 1.5 * delta
            upper_clip_value = upper_quantile + 1.5 * delta
            clipped_weights = np.clip(layer_weights, lower_clip_value, upper_clip_value)
            result.append(np.mean(clipped_weights, axis=0).tolist())
        return result
    
    def fixed_clip_aggregation(self, model_weights):
        result = []
        for layer_index in range(len(model_weights[0])):
            layer_weights = np.array(
                [
                    single_model_weights[layer_index]
                    for single_model_weights in model_weights
                ]
            )
            # weight_range = 0.5 - (-0.5)  # Range of typical weights
            # fraction_of_range = 0.1  
            # clip_threshold = fraction_of_range * weight_range
            clipped_weights = np.clip(layer_weights, -1, 1)
            result.append(np.mean(clipped_weights, axis=0).tolist()  )
        return result
    
    def weighted_score_aggregation(self, score_weights, model_weights):
        result = []
        for layer_index in range(len(model_weights[0])):
            layer_weights = np.array(
                [
                    single_model_weights[layer_index]
                    for single_model_weights in model_weights
                ]
            )
            result.append(np.average(layer_weights, axis=0, weights=score_weights.ravel()).tolist())
        return result

    def weighted_recency_aggregation(self, recency_weights, model_weights):
        result = []
        for layer_index in range(len(model_weights[0])):
            layer_weights = np.array(
                [
                    single_model_weights[layer_index]
                    for single_model_weights in model_weights
                ]
            )
            result.append(np.average(layer_weights, axis=0, weights=recency_weights).tolist())
        return result
    
    def weighted_punish_aggregation(self, punish_weights, model_weights):
        result = []
        for layer_index in range(len(model_weights[0])):
            layer_weights = np.array(
                [
                    single_model_weights[layer_index]
                    for single_model_weights in model_weights
                ]
            )
            result.append(np.average(layer_weights, axis=0, weights=np.full(len(layer_weights), punish_weights)).tolist())
        return result

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
                logger.info("Cutting of old data")
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
                new_config = self.gradual_means_hp_update(hp_values, hp_names)
            elif self.meta_learning_mode == "GRADUAL_SCORE_WEIGHTED_MEANS":
                new_config = self.gradual_score_weighted_means_hp_update(
                    hp_values, hp_names, scores
                )
            elif self.meta_learning_mode == "DIRECT_GENETIC_ALGORITHM_MATING":
                new_config = self.genetic_algorithm_mating_hp_update(
                    hp_values, hp_names, scores, update_rate=1.0
                )
            elif self.meta_learning_mode == "GRADUAL_GENETIC_ALGORITHM_MATING":
                new_config = self.genetic_algorithm_mating_hp_update(
                    hp_values, hp_names, scores
                )
            elif self.meta_learning_mode == "NO_META_LEARNING":
                new_config = self.hp_config
            else:
                test=self.meta_learning_mode == "NO_META_LEARNING"
                logger.info(f"Test result is {test}")
                raise ValueError(
                    f"Invalid value for meta_learning_mode: \
                        {self.meta_learning_mode} was given."
                )
            logger.info(f"New config: {new_config}")
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
            population = np.random.choice(
                hp_values[name], size=500, p=np.array(scores) / sum(scores)
            )
            new_config[name + "_mean"] = (1 - update_rate) * self.hp_config[
                name + "_mean"
            ] + update_rate * np.average(population)
            new_config[name + "_std"] = (1 - update_rate) * self.hp_config[
                name + "_std"
            ] + update_rate * np.std(population)
        return new_config

    def log_step(self):
        log_data = deepcopy(self.hp_config)
        for key in self.result_container.keys():
            log_data[key] = self.result_container[key]
        log_data["weights"] = self.weights
        log_data["timestamp"] = datetime.now()
        pickle.dump(
            log_data,
            open(LOG_PATH / f"parameter_server/{str(time.time())}.pkl", mode="wb"),
        )

    # def update_hp_config(self) -> None:
    #     """Method to update the hp_config that is given the nodes as
    #     baseline to generate its hyper_parameters from.
    #     """
    #     hp_names = set(["_".join(key.split("_")[:-1]) for key in self.hp_config.keys()])
    #     hp_values = dict()
    #     scores = []
    #     for name in hp_names:
    #         hp_values[name] = []
    #     update_counter = 0
    #     for node_id, node_results in self.result_container.items():
    #         for name in hp_names:
    #             if name == "learning_rate":
    #                 hp_values[name] += [
    #                     np.log10(value[name]) for value in node_results["hp_params"]
    #                 ]
    #             else:
    #                 hp_values[name] += [
    #                     value[name] for value in node_results["hp_params"]
    #                 ]
    #         update_counter += len(node_results["hp_params"])
    #         scores += node_results["scores"]
    #     if update_counter >= 5:
    #         if self.meta_learning_mode == "DIRECT_MEANS":
    #             new_config = self.gradual_means_hp_update(
    #                 hp_values, hp_names, update_rate=1.0
    #             )
    #         elif self.meta_learning_mode == "DIRECT_SCORE_WEIGHTED_MEANS":
    #             new_config = self.gradual_score_weighted_means_hp_update(
    #                 hp_values, hp_names, scores, update_rate=1.0
    #             )
    #         elif self.meta_learning_mode == "GRADUAL_MEANS":
    #             new_config = self.gradual_means_hp_update(hp_values, hp_names, scores)
    #         elif self.meta_learning_mode == "GRADUAL_SCORE_WEIGHTED_MEANS":
    #             new_config = self.gradual_score_weighted_means_hp_update(
    #                 hp_values, hp_names, scores
    #             )
    #         elif self.meta_learning_mode == "DIRECT_GENETIC_ALGORITHM_MATING":
    #             new_config = self.genetic_algorithm_mating_hp_update(
    #                 hp_values, hp_names, scores, update_rate=1.0
    #             )
    #         elif self.meta_learning_mode == "GRADUAL_GENETIC_ALGORITHM_MATING":
    #             new_config = self.genetic_algorithm_mating_hp_update(
    #                 hp_values, hp_names, scores
    #             )
    #         else:
    #             raise ValueError(
    #                 f"Invalid value for meta_learning_mode: \
    #                     {self.meta_learning_mode} was given."
    #             )
    #         logger.info(f"New config: {new_config}")
    #         self.hp_config = new_config

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
            population = np.random.choice(
                hp_values[name], size=500, p=np.array(scores) / sum(scores)
            )
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
