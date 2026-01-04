"""
Prompts for World Model (Perception and Prediction).
Aligned with paper specifications (Appendix prompts).
"""

# System prompts (not shown in paper, kept minimal)
PERCEPTION_SYSTEM_PROMPT = """You are a world model that perceives and tracks the state of a household environment."""

PREDICTION_SYSTEM_PROMPT = """You are a world model that predicts the state of a household environment after an action."""


# Perception prompt template (from paper Appendix)
PERCEPTION_PROMPT_TEMPLATE = """[PER]
You are a world model that perceives and tracks the state of a household environment.

State of this environment is described as a dictionary with the following keys:
- "agent_location": the location of the agent in the environment; 
- "inventory": the object the agent is holding (if any);
- "world_state": a dictionary describing the state of each receptacle in the room, with the following keys:
    - "visited": whether the agent has examined the objects in/on the receptacle;
    - "open": whether the receptacle is open; This is only applicable to openable receptacles;
    - "objects": a list of objects in the receptacle. 

Now you will be given the following information:
- action: an action that the agent has executed in the environment;
- action_success: whether the action was successfully executed;
- old_state: the state of the environment before the action;
- new_frame: the first-person view of the environment after the action;
- other_visible_receptacles: for "go to" actions, you will also be given the names of other visible receptacles in the new frame, and you should update the object list of these receptacles as well.

Your task is to update and return the state of the environment in the right format based on the above information. No verbal explanation is needed.

action: {action}
action_success: {action_success}
old_state: {old_state}
other_visible_receptacles: {other_visible_receptacles}

The new_frame is provided right after the textual prompt."""


# Prediction prompt template with MetaICL (from paper Appendix)
PREDICTION_PROMPT_TEMPLATE_METAICL = """[PRE]
You are a world model that predicts the state of a household environment after an action. 

State of this environment is described as a dictionary with the following keys:
- "agent_location": the location of the agent in the environment; 
- "inventory": the object the agent is holding (if any);
- "world_state": a dictionary describing the state of each receptacle in the room, with the following keys:
    - "visited": whether the agent has examined the objects in/on the receptacle; 
    - "open": whether the receptacle is open; this is only applicable to openable receptacles; 
    - "objects": a list of objects in the receptacle. 

You are given the following information:
- action: an action that the agent will execute in the environment;
- old_state: the state of the environment before the action. 

Your task is to predict the state of the environment after the action and return it in the right format. No verbal explanation is needed.

action: {action}
old_state: {old_state}

Here are observations from previous trajectories (receptacles and object lists): 
{condition_1}

Here are failed actions from previous trajectories (action, agent location, inventory, state of the target receptacle): 
{condition_2}"""
