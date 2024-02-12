import json
from datetime import datetime
from itertools import chain
from typing import Dict

import numpy as np
from mqtt_builder import get_mqqt_client


class Runner:
    def __init__(self, aggregation_method: str) -> None:
        self.aggregation_method = aggregation_method
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
        print("Got the payload: ", payload)
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
        self.maintain_result_container(id,payload)
        model_weights = []
        for node_id, node_results in self.result_container.items():
            model_weights += node_results["weights"]
        global_model = []

        if(self.aggregation_method=="AVERAGE AGGREGATION"):
            global_model=self.average_aggregation(model_weights)
        elif(self.aggregation_method=="FIXED CLIPPED AVERAGE AGGREGATION"):
            global_model=self.median_aggregation(model_weights)
        elif(self.aggregation_method=="DYNAMIC CLIPPED AVERAGE AGGREGATION"):
            global_model=self.dynamic_clip_aggregation(model_weights)
        elif(self.aggregation_method=="MEADIAN AGGREGATION"):
            global_model=self.fixed_clip_aggregation(model_weights)
        elif(self.aggregation_method=="WEIGHTED AVERAGE - PERFORMANCE SCORES"):
            model_scores = []
            model_scores = [node_results["scores"] for node_results in self.result_container.values()]
            score_weights = np.exp(model_scores) / np.sum(np.exp(model_scores))
            global_model=self.weighted_score_aggregation(score_weights, model_weights)
        elif(self.aggregation_method=="WEIGHTED AVERAGE - RECENCY"):
            decay_rate = 0.9
            timestamps = chain.from_iterable([node_results["timestamps"] for node_results in self.result_container.values()])
            time_diff_seconds = np.array([(datetime.now() - timestamp).total_seconds() for timestamp in timestamps])
            recency_weights = np.exp(-decay_rate * time_diff_seconds)
            global_model=self.weighted_recency_aggregation(recency_weights,model_weights)
                
      
        # for layer_index in range(len(model_weights[0])):
        #     layer_weights = np.array(
        #         [
        #             single_model_weights[layer_index]
        #             for single_model_weights in model_weights
        #         ]
        #     )
            # print(layer_weights.shape)
            # global_model.append(self.average_aggregation(layer_weights))
            # global_model.append(np.mean(layer_weights, axis=0).tolist())
            # global_model.append(self.median_aggregation(layer_weights))
            # global_model.append(self.fixed_clip_aggregation(layer_weights))
            # global_model.append(self.dynamic_clip_aggregation(layer_weights))
            # weighted average - performance scores
            # global_model.append(np.average(layer_weights, axis=0, weights=score_weights.ravel()).tolist())
            # weighted average - recency 
            # global_model.append(np.average(layer_weights, axis=0, weights=recency_weights).tolist())

        self.weights = global_model  # payload["best_model_weights"] # Temporary test
            

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
            weight_range = 0.5 - (-0.5)  # Range of typical weights
            fraction_of_range = 0.1  
            clip_threshold = fraction_of_range * weight_range
            clipped_weights = np.clip(layer_weights, -clip_threshold, clip_threshold)
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


    def maintain_result_container(self, id, payload)->None:
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
