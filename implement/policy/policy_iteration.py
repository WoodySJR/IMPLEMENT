"""
Policy Iteration Module for IMPLEMENT.
Implements the model-based online policy iteration algorithm.
"""

import time
from typing import List, Dict, Any, Tuple

from config import config
from implement.prompts.policy_prompts import (
    POLICY_SYSTEM_PROMPT,
    POLICY_PROMPT_TEMPLATE,
    BELIEF_STATE_PROMPT_TEMPLATE,
    LAST_PROPOSAL_PROMPT
)
from implement.utils.llm_utils import get_policy_response, get_belief_state_response
from implement.world_model.prediction import world_model_prediction_metaicl


def get_policy_response_with_wm(
    goals: List[str],
    current_states: List[Dict[str, Any]],
    histories: List[List[str]],
    tasks: List[str],
    model: str,
    conditions_1: List[List[Any]],
    conditions_2: List[List[Any]]
) -> List[str]:
    """
    Get policy responses using model-based policy iteration.
    
    Args:
        goals: List of task goals
        current_states: List of current states
        histories: List of action histories
        tasks: List of task identifiers
        model: Policy model name
        conditions_1: Historical observations for MetaICL
        conditions_2: Historical failed actions for MetaICL
        
    Returns:
        List of actions to execute
    """
    print(f"Policy iteration with histories: {histories}")
    
    active_envs = list(range(len(goals)))  # Indices of environments not ready yet
    actions_ready = {}  # Actions ready to execute
    previous_actions = {}  # Actions proposed in previous rounds
    for i in active_envs:
        previous_actions[i] = []
    
    total_rounds = 0
    
    # Policy iteration loop
    while len(active_envs) > 0 and total_rounds < config.policy.max_interaction_rounds:
        print(f"Round {total_rounds + 1}, previous actions: {previous_actions}")
        
        # Step 1: Action sampling from policy LLM
        active_prompts = []
        for i in active_envs:
            prompt = POLICY_PROMPT_TEMPLATE.format(
                goal=goals[i],
                current_state=current_states[i],
                history="\n".join(histories[i]),
                evaluations=previous_actions[i]
            )
            active_prompts.append(prompt)
        
        start_time = time.time()
        responses = get_policy_response(
            prompts=active_prompts,
            model=model,
            system_prompt=POLICY_SYSTEM_PROMPT,
            temperature=config.llm.temperature
        )
        end_time = time.time()
        print(f"Action sampling time: {end_time - start_time:.2f} seconds")
        print(f"Responses: {responses}")
        
        # Parse actions and help flags
        actions = []
        for response in responses:
            try:
                actions.append(response.split("<action>")[1].split("</action>")[0].strip())
            except:
                actions.append("")
        print(f"Actions: {actions}")
        
        helps = []
        for response in responses:
            try:
                helps.append(response.split("<help>")[1].split("</help>")[0].strip())
            except:
                print(f"Error parsing help flag: {response}")
                helps.append("True")
        print(f"Help flags: {helps}")
        
        # Force help=True for new actions
        for ii, i in enumerate(active_envs):
            if len(previous_actions[i]) == 0 or actions[ii] not in [action["action"] for action in previous_actions[i]]:
                helps[ii] = "True"
        
        # Remove environments that are ready to execute
        to_remove = []
        for ii, i in enumerate(active_envs):
            if "False" in helps[ii]:
                to_remove.append(i)
                actions_ready[i] = actions[ii]
            else:
                previous_actions[i].append({"action": actions[ii], "prediction": None})
        
        for i in to_remove:
            active_envs.remove(i)
        
        # Step 2: World model prediction for new actions
        if len(active_envs) > 0:
            samples = []
            for i in active_envs:
                action = previous_actions[i][-1]["action"]
                info_gather = False
                if "go to" in action or "open" in action:
                    target_loc = " ".join(action.split(" ")[-2:])
                    if target_loc in current_states[i]["world_state"] and not current_states[i]["world_state"][target_loc]["visited"]:
                        info_gather = True
                
                sample = {
                    "action": action,
                    "old_state": current_states[i],
                    "info_gather": info_gather,
                    "condition_1": conditions_1[i],
                    "condition_2": conditions_2[i]
                }
                samples.append(sample)
            
            start_time = time.time()
            predictions = world_model_prediction_metaicl(samples)
            end_time = time.time()
            print(f"Prediction time: {end_time - start_time:.2f} seconds")
            print(f"Predictions: {predictions}")
            
            # Step 3: Belief state summarization
            start_time = time.time()
            belief_states = get_belief_state(
                predictions,
                [current_states[i] for i in active_envs],
                [previous_actions[i][-1]["action"] for i in active_envs],
                config.llm.belief_state_model
            )
            end_time = time.time()
            print(f"Belief state summarization time: {end_time - start_time:.2f} seconds")
            print(f"Belief states: {belief_states}")
            
            # Update previous actions with belief states
            for ii, i in enumerate(active_envs):
                previous_actions[i][-1]["prediction"] = belief_states[ii]
        
        total_rounds += 1
    
    # Handle remaining environments (max rounds reached)
    if len(active_envs) > 0:
        active_prompts = []
        for i in active_envs:
            prompt = POLICY_PROMPT_TEMPLATE.format(
                goal=goals[i],
                current_state=current_states[i],
                history="\n".join(histories[i]),
                evaluations=previous_actions[i]
            )
            active_prompts.append(prompt + LAST_PROPOSAL_PROMPT)
        
        responses = get_policy_response(
            prompts=active_prompts,
            model=model,
            system_prompt=POLICY_SYSTEM_PROMPT,
            temperature=config.llm.temperature
        )
        
        actions = []
        for response in responses:
            try:
                actions.append(response.split("<action>")[1].split("</action>")[0].strip())
            except:
                actions.append("")
        print(f"Final actions: {actions}")
        
        for ii, i in enumerate(active_envs):
            actions_ready[i] = actions[ii]
    
    return [actions_ready[i] for i in range(len(goals))]


def get_belief_state(
    predictions: List[List[str]],
    current_states: List[Dict[str, Any]],
    actions: List[str],
    model: str
) -> List[str]:
    """
    Summarize Monte Carlo predictions into belief states.
    
    Args:
        predictions: List of lists of predicted states
        current_states: List of current states
        actions: List of actions
        model: Model for summarization
        
    Returns:
        List of belief state strings
    """
    prompts = []
    for prediction, current_state, action in zip(predictions, current_states, actions):
        prompt = BELIEF_STATE_PROMPT_TEMPLATE.format(
            predicted_future_states=prediction,
            current_state=current_state,
            action=action
        )
        prompts.append(prompt)
    
    outputs = get_belief_state_response(
        prompts=prompts,
        model=model,
        temperature=config.llm.temperature
    )
    
    return outputs

