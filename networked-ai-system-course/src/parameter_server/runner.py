import json
from datetime import datetime
from typing import Dict
import sys
import numpy as np
from mqtt_builder import get_mqqt_client
from loguru import logger


logger.add(sys.stderr, format="{time} - {level} - {message}", level="DEBUG")
# logging.basicConfig(
#     format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.DEBUG
# )
# logger = logging.getLogger(__name__)


class Runner:
    def __init__(self) -> None:
        self.mqtt_client = get_mqqt_client("server", self.respond)
        self.weights: None | np.ndarray = None
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
        self.result_container: Dict = {}

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
            logger.info("Trying to aggregate...")
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
            if n_datapoints > 3:
                for key, value in node_results.items():
                    node_results[key] = value[-3:]
            self.result_container[node_id] = node_results

    def run(self) -> None:
        """Main method causing the runner to have its behaviour.
        For this runner, the behaviour is listening to mqtt messages and
        react on them. Therefore, run() only calls the loop_forever() method
        of the mqtt client.
        """
        self.mqtt_client.loop_forever()
