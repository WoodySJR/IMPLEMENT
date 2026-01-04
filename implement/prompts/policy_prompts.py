"""
Prompts for Policy LLM.
Aligned with paper specifications (Appendix prompts).
"""

# System prompt (not shown in paper, kept minimal)
POLICY_SYSTEM_PROMPT = """You are a text adventure game player, who interacts with a household environment to complete a task."""


# Main policy prompt template (from paper Appendix)
POLICY_PROMPT_TEMPLATE = """You are a text adventure game player, who interacts with a household environment to complete a task. 

The following is a list of actions that can be used in the game. You must strictly follow the syntax of the actions in your answer. 
1. go to (receptacle): move to a receptacle
2. open (receptacle): open a receptacle
3. close (receptacle): close a receptacle
4. take (object) from (receptacle): take an object from a receptacle
5. put (object) to (receptacle): place an object in your inventory into or onto a receptacle
6. use (object): turn on the object, which is typically a light source like a desklamp or a floorlamp
7. heat (object) with (receptacle): heat an object with a receptacle. 
8. cool (object) with (receptacle): cool an object with a receptacle.
9. clean (object) with (receptacle): clean an object with a receptacle.

The following is your task goal. 
{goal}

The following is your history interactions. 
{history}

The following is the current state of the game. Specifically, the state of the environment is described as a dictionary with the following keys:
- "agent_location": your current location in the environment; 
- "inventory": the object you are holding (if any);
- "world_state": a dictionary describing the state of each receptacle in the room, with the following keys:
    - "visited": whether you have examined the objects in/on the receptacle; 
    - "open": whether the receptacle is open; This is only applicable to openable receptacles;
    - "objects": a list of objects in the receptacle. 
Current state: {current_state}

A world model is available that can predict the state after an action is executed. Here are a few actions, along with their predicted future states: 
{evaluations}
When proposing actions, you have only two options:
Option 1. Choose an action from the above provided actions to execute (must output "False" in "help").
    - This option is available only when any of the provided actions' predicted future state is helpful for completing the task. 
    - This option is not available when there is no action provided above. 
    - You must EXPLICITLY rank the helpfulness of the above provided actions according to their predicted future states in your <think> section, and then output the most promising one in <action>.  
Option 2. Propose up to 8 different actions (must output "True" in "help").
    - This option is feasible when none of the above provided actions' predicted future state is helpful enough for completing the task. 
    - In this case, you should propose up to 8 DIFFERENT actions that are promising and worth trying. 
    - The actions you propose will not be executed. Instead, their predicted future states will be returned to you to evaluate their helpfulness. 

Your answer must be formatted as: 
<think>
your reasoning process
</think>
<action>
action proposal(s) separated by commas
</action>
<help>
whether need world model to evaluate the action(s)
</help>."""


# Last proposal prompt (when max rounds reached)
LAST_PROPOSAL_PROMPT = """

PAY ATTENTION!! This is the last chance to propose an action. Only "option 1" is available, which means you must choose the best action from the provided actions."""


# Belief state summarization prompt (from paper Appendix)
BELIEF_STATE_PROMPT_TEMPLATE = """A text adventure game player is interacting with a household environment to complete a task. The following is the current state of the environment. Specifically, the state of the environment is described as a dictionary with the following keys:
- "agent_location": the location of the agent in the environment; 
- "inventory": the object the agent is holding (if any);
- "world_state": a dictionary describing the state of each receptacle in the room, with the following keys:
    - "visited": whether the agent has examined the objects in/on the receptacle;
    - "open": whether the receptacle is open; This is only applicable to openable receptacles (drawer, cabinet, fridge, microwave, safe, etc.);
    - "objects": a list of objects in the receptacle.

Current state: {current_state}

Here is an action that the agent has proposed. 
Action: {action}

The following are one or more possible future states of the environment after the action is executed, predicted by a world model. 
Your task is to summarize the predicted future states into a coherent belief state that better aids the game player's decision. 
You should focus on and clearly describe the changes in the future state compared to the current state (such as the locations of agent and objects, the states of objects and receptacles, etc.), and how likely the changes are to happen (based on their frequencies in the predicted results). 
Importantly, your summary must be neutral and objective, without any suggestions or speculations beyond the predicted results, or any judgment regarding the helpfulness of the action. Limit your answer to within 150 words. 

Predicted future states: {predicted_future_states}"""
