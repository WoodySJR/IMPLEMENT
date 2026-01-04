"""
State representation and manipulation utilities for IMPLEMENT.
Handles conversion between observations and symbolic states.
"""

import copy
from typing import Dict, List, Any


def get_all_receptacles(obs: str) -> List[str]:
    """
    Extract all receptacles from initial observation.
    
    Args:
        obs: Initial observation string
        
    Returns:
        List of receptacle names
    """
    receps = obs.split("you see")[1].replace(" a ", "").replace(" and", "").replace(".", "").split("\n\nYour")[0].split(",")
    return [receptacle.strip() for receptacle in receps]


def initial_obs_to_state(obs: str) -> Dict[str, Any]:
    """
    Convert initial observation to symbolic state representation.
    
    Args:
        obs: Initial observation string
        
    Returns:
        State dictionary with keys: agent_location, inventory, world_state
    """
    state = {}
    state["agent_location"] = "middle of the room"
    state["inventory"] = None
    state["world_state"] = {}

    all_receptacles = get_all_receptacles(obs)
    for receptacle in all_receptacles:
        if "drawer" in receptacle or "cabinet" in receptacle:
            state["world_state"][receptacle] = {"visited": False, "open": False, "objects": []}
        else:
            state["world_state"][receptacle] = {"visited": False, "open": "N/A", "objects": []}

    return state


def obs_to_state(state: Dict[str, Any], obs: str, action: str) -> Dict[str, Any]:
    """
    Update state based on observation and action.
    
    Args:
        state: Current state dictionary
        obs: Observation string after action
        action: Action that was executed
        
    Returns:
        Updated state dictionary
    """
    obs = obs.replace(" the", "")
    state = copy.deepcopy(state)

    # If nothing happens, return the same state
    if "Nothing happens" in obs:
        return state
    
    # Agent location change and world state update
    if "go to" in action and "Nothing happens" not in obs:
        state["agent_location"] = action.split("go to")[1].strip()
        if "you see" in obs and "nothing" not in obs:  # Open receptacle with objects
            state["world_state"][state["agent_location"]]["visited"] = True
            if "drawer" in state["agent_location"] or "cabinet" in state["agent_location"]:
                state["world_state"][state["agent_location"]]["open"] = True
            else:
                state["world_state"][state["agent_location"]]["open"] = "N/A"
            if state["world_state"][state["agent_location"]]["objects"] is None:
                state["world_state"][state["agent_location"]]["objects"] = obs.split("you see")[1].replace(" a ", "").replace(" and", "").replace(".", "").split(",")
            else:
                state["world_state"][state["agent_location"]]["objects"].extend(obs.split("you see")[1].replace(" a ", "").replace(" and", "").replace(".", "").split(","))
                state["world_state"][state["agent_location"]]["objects"] = list(set(state["world_state"][state["agent_location"]]["objects"]))
        if "you see" in obs and "nothing" in obs:  # Open receptacle with no objects
            state["world_state"][state["agent_location"]]["visited"] = True
            if "drawer" in state["agent_location"] or "cabinet" in state["agent_location"]:
                state["world_state"][state["agent_location"]]["open"] = True
            else:
                state["world_state"][state["agent_location"]]["open"] = "N/A"
            if state["world_state"][state["agent_location"]]["objects"] is None:
                state["world_state"][state["agent_location"]]["objects"] = []
            else:
                state["world_state"][state["agent_location"]]["objects"].extend([])
        if "closed" in obs:  # Closed receptacle
            state["world_state"][state["agent_location"]]["open"] = False
    
    # Open a receptacle
    if "You open" in obs:
        receptacle = obs.split("open")[1].split(".")[0].strip()
        state["world_state"][receptacle]["visited"] = True
        state["world_state"][receptacle]["open"] = True
        if "nothing" in obs:
            state["world_state"][receptacle]["objects"] = []
        else:
            state["world_state"][receptacle]["objects"] = obs.split("you see")[1].replace(" a ", "").replace(" and", "").replace(".", "").split(",")
    
    # Close a receptacle
    if "You close" in obs:
        receptacle = obs.split("close")[1].split(".")[0].strip()
        state["world_state"][receptacle]["open"] = False

    # Take an object
    if "You pick up" in obs:
        item = obs.split("pick up")[1].split("from")[0].strip().replace(".", "")
        source = obs.split("pick up")[1].split("from")[1].strip().replace(".", "")
        for obj in state["world_state"][source]["objects"]:
            if item in obj:
                state["world_state"][source]["objects"].remove(obj)
                state["inventory"] = obj
                break

    # Put an object
    if "You put" in obs:
        destination = obs.split("You put")[1].split(" to ")[1].replace(".", "").strip()
        state["world_state"][destination]["objects"].append(state["inventory"])
        state["inventory"] = None

    # Heat an object
    if "heat" in action:
        item = action.split("heat")[1].split("with")[0].strip()
        state["inventory"] = item + "(hot)"

    # Cool an object
    if "cool" in action:
        item = action.split("cool")[1].split("with")[0].strip()
        state["inventory"] = item + "(cool)"

    # Clean an object
    if "clean" in action:
        item = action.split("clean")[1].split("with")[0].strip()
        state["inventory"] = item + "(clean)"

    # Turn on lamp
    if "use" in action and "lamp" in action:
        lamp = action.split("use")[1].strip()
        receptacle = state["agent_location"]
        for obj in state["world_state"][receptacle]["objects"]:
            if lamp in obj:
                state["world_state"][receptacle]["objects"].remove(obj)
                state["world_state"][receptacle]["objects"].append(obj + "(on)")
                break

    return state
