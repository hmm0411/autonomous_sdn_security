import requests
import numpy as np
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ControllerClient:
    def __init__(
        self, 
        onos_ip: str = "34.126.64.185", 
        port: str = "8181", 
        user: str = "onos", 
        pwd: str = "rocks", 
        use_mock: bool = True,
        timeout: int = 10
    ):
        self.controller_url = f"http://{onos_ip}:{port}"
        self.auth = (user, pwd)
        self.use_mock = use_mock
        self.timeout = timeout
        self.last_action = 0

    def get_state(self) -> Dict[str, float]:
        """
        Get current network state from controller
        Returns normalized state dict with 10 features matching PPO agent expectations
        """
        if self.use_mock:
            return self._get_mock_state()
        
        try:
            response = requests.get(
                f"{self.controller_url}/state",
                auth=self.auth,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get state from controller: {e}")
            return self._get_mock_state()

    def _get_mock_state(self) -> Dict[str, float]:
        """Generate mock state data for testing"""
        return {
            "packet_rate": float(np.random.uniform(100, 1000)),
            "byte_rate": float(np.random.uniform(1000, 10000)),
            "flow_count": float(np.random.uniform(10, 100)),
            "src_ip_entropy": float(np.random.uniform(0.0, 1.0)),
            "latency": float(np.random.uniform(0.0, 1.0)),
            "packet_loss": float(np.random.uniform(0.0, 1.0)),
            "queue_length": float(np.random.uniform(0.0, 1.0)),
            "controller_cpu": float(np.random.uniform(0.0, 1.0)),
            "attack_indicator": float(np.random.uniform(0.0, 1.0)),
            "previous_action": float(self.last_action)
        }

    def apply_action(self, action: int) -> Dict[str, Any]:
        """
        Apply an action to the network
        
        Args:
            action: Action index (0-4)
        
        Returns:
            Response dict with status information
        """
        if not isinstance(action, int) or action < 0 or action > 4:
            logger.warning(f"Invalid action: {action}. Using no_action (0)")
            action = 0
        
        self.last_action = action
        
        if self.use_mock:
            return {
                "status": "success",
                "action": action,
                "message": f"Mock action applied: {action}"
            }
        
        try:
            response = requests.post(
                f"{self.controller_url}/apply_action",
                json={"action": action},
                auth=self.auth,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to apply action to controller: {e}")
            return {
                "status": "error",
                "action": action,
                "message": str(e)
            }

    def reset(self) -> None:
        """Reset the client state"""
        self.last_action = 0
        logger.info("Controller client reset")