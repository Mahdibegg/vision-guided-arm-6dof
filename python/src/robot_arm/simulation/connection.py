from dataclasses import dataclass
from typing import Any
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

from config import load_simulation_config

SIMULATION_CONFIG = load_simulation_config()

host = SIMULATION_CONFIG.host
port = SIMULATION_CONFIG.port

@dataclass(slots=True)
class SimulationConnection:
    """
    A single dedicated ZMQ Remote API connection to CoppeliaSim.
 
    A ZMQ REQ socket is not safe to share across threads - calling sim.*
    from two threads on the same connection can desync the socket and raise
    "operation cannot be done in current state" errors. Each thread that
    calls into sim (the GUI thread, the camera thread, the robot arm thread,
    etc.) should hold its own SimulationConnection, created via
    connect_to_simulation() on that thread, rather than being handed one
    that was created elsewhere.
    """
    client: RemoteAPIClient
    sim: Any
 
    def is_simulation_running(self) -> bool:
        """Checks whether the simulation is actively advancing (not stopped/paused)."""
        try:
            state = self.sim.getSimulationState()
 
            # 0: stopped, 1: paused, >=2: advancing/running
            stopped = getattr(self.sim, "simulation_stopped", 0)
            paused = getattr(self.sim, "simulation_paused", 1)
 
            return state not in (stopped, paused)
 
        except Exception as e:
            # Print the error to terminal so it doesn't fail silently
            print(f"Simulation state check failed: {e}")
            return False

def connect_to_simulation() -> SimulationConnection:
    """
    Establishes a new, independent connection to the CoppeliaSim ZeroMQ Remote API.
    Call this once per thread that needs to talk to the simulator - never
    share the returned SimulationConnection across threads.
    Returns a SimulationConnection, or raises a descriptive error.
    """
    try:
        client = RemoteAPIClient(SIMULATION_CONFIG.host, SIMULATION_CONFIG.port)
        sim = client.require('sim')
        return SimulationConnection(client, sim)
 
    except Exception as e:
        raise ConnectionError(
            f"Failed to connect to CoppeliaSim at {SIMULATION_CONFIG.host}:{SIMULATION_CONFIG.port}. "
            "Please ensure the simulator is open and the ZeroMQ plugin is loaded."
        ) from e