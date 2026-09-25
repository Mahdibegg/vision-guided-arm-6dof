from typing import Any
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from arm.config_loader import load_simulation_config

# Load simulation connection details to establish a connection when needed

SIMULATION_CONFIG = load_simulation_config()

host = SIMULATION_CONFIG.host
port = SIMULATION_CONFIG.port

def connect_to_simulation() -> tuple[RemoteAPIClient, Any]:
    """
    Establishes a connection to the CoppeliaSim ZeroMQ Remote API.
    Returns the client and the sim object, or raises a descriptive error.
    """
    try:
        client = RemoteAPIClient(host, port)
        sim = client.require('sim')
        return client, sim
    except Exception as e:
        raise ConnectionError(
            f"Failed to connect to CoppeliaSim at {host}:{port}. "
            "Please ensure the simulator is open and the ZeroMQ plugin is loaded."
        ) from e