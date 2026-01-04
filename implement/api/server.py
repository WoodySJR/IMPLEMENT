"""
FastAPI Server for IMPLEMENT.
Provides API endpoints for policy generation and world model inference.
"""

from fastapi import FastAPI
import uvicorn
from typing import Dict, Any, List

from config import config
from implement.policy.policy_iteration import get_policy_response_with_wm
from implement.world_model.perception import world_model_perception


app = FastAPI(title="IMPLEMENT API", version="1.0.0")


@app.post("/generate_policy")
async def generate_policy(params: Dict[str, Any]) -> List[str]:
    """
    Generate policy actions using model-based policy iteration.
    
    Args:
        params: Dictionary containing:
            - current_goals: List of task goals
            - current_states: List of current states
            - histories: List of action histories
            - current_tasks: List of task identifiers
            - model: Policy model name
            - conditions_1: Historical observations for MetaICL
            - conditions_2: Historical failed actions for MetaICL
            
    Returns:
        List of actions to execute
    """
    current_goals = params["current_goals"]
    current_states = params["current_states"]
    model = params["model"]
    histories = params["histories"]
    tasks = params["current_tasks"]
    conditions_1 = params["conditions_1"]
    conditions_2 = params["conditions_2"]
    
    actions = get_policy_response_with_wm(
        goals=current_goals,
        current_states=current_states,
        histories=histories,
        tasks=tasks,
        model=model,
        conditions_1=conditions_1,
        conditions_2=conditions_2
    )
    
    return actions


@app.post("/perception")
async def perception(params: Dict[str, Any]) -> List[str]:
    """
    Perform world model perception.
    
    Args:
        params: Dictionary containing:
            - samples: List of perception samples
            
    Returns:
        List of perceived states
    """
    states = world_model_perception(params["samples"])
    return states


@app.post("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}


def start_server(host: str = None, port: int = None):
    """
    Start the FastAPI server.
    
    Args:
        host: Server host (default from config)
        port: Server port (default from config)
    """
    host = host or config.api.host
    port = port or config.api.port
    
    print(f"Starting IMPLEMENT API server at {host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_server()

