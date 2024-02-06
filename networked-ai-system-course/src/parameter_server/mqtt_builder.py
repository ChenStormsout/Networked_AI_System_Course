import time

import paho.mqtt.client as mqtt
import os

HOST = os.getenv('mqtt_host')
if HOST is None:
    HOST="0.0.0.0"
print(HOST)

def get_mqqt_client(
    client_id: str,
    on_message_func: callable,
    host: str = HOST,
    port: int = 1883,
    connection_retries: int = 3,
) -> mqtt.Client:
    """Creates and connects a mqtt client that then can be used
    for sending and receiving messages.

    Parameters
    ----------
    client_id : str
        ID of the mqtt client
    on_message_func: callable
        function or method that shall be called when the client recieves
        a message
    host : str, optional
        Host IP of the Mqtt message broker, by default "0.0.0.0"
    port : int, optional
        Port of the Mqtt message broker, by default 1883
    connection_retries : int, optional
        Number of retries for a connection attempt, by default 3

    Returns
    -------
    mqtt.Client
        Connected Mqtt client.

    Raises
    ------
    ConnectionRefusedError
        In case it is not possible to connect to the Mqtt
        broker, this exception is raised.
    """
    mqtt_client = mqtt.Client(client_id=client_id)
    mqtt_client.on_message = on_message_func
    response_code = None
    for i in range(connection_retries):
        print(f"Trying to connect to mqtt server. Attempt: {i+1}")
        try:
            response_code = mqtt_client.connect(host=host, port=port)
            if response_code == 0:
                print("Connection succeded")
                break
        except Exception as error:
            print("Connection failed")
            time.sleep(1)
            if i == connection_retries - 1:
                raise ConnectionRefusedError(
                    error,
                    f"Initial connection to MQTT server failed for client\
                    {client_id} after {connection_retries} attempts.\
                    Response code: ",
                    response_code,
                )
    mqtt_client.subscribe("server_update")
    mqtt_client.subscribe("server_get_model")
    return mqtt_client
