"""
World Model Prediction Module.
Handles state prediction with MetaICL for test-time adaptation.
"""

import base64
import random
import copy
import time
from typing import List, Dict, Any, Tuple
from openai import OpenAI
from tqdm import tqdm

from config import config
from implement.prompts.world_model_prompts import (
    PREDICTION_SYSTEM_PROMPT,
    PREDICTION_PROMPT_TEMPLATE_METAICL
)
from implement.utils.llm_utils import get_vllm_response


# Initialize vLLM client (same as perception - unified world model)
vllm_client = OpenAI(
    base_url=f"http://{config.world_model.vllm_host}:{config.world_model.vllm_port}/v1",
    api_key=config.world_model.vllm_api_key
)

# Load placeholder black image
BLACK_IMAGE_PATH = "./assets/black.png"
try:
    base64_black = base64.b64encode(open(BLACK_IMAGE_PATH, "rb").read()).decode('utf-8')
except FileNotFoundError:
    print(f"Warning: Black placeholder image not found at {BLACK_IMAGE_PATH}")
    base64_black = ""


def world_model_prediction_metaicl(samples: List[Dict[str, Any]]) -> List[List[str]]:
    """
    Perform state prediction with MetaICL conditioning.
    Uses Monte Carlo rollouts for information-gathering actions.
    
    Args:
        samples: List of sample dictionaries, each containing:
            - action: Action to predict
            - old_state: Current state
            - info_gather: Whether this is an info-gathering action
            - condition_1: Historical observations (receptacles and objects)
            - condition_2: Historical failed actions
            
    Returns:
        List of lists of predicted states (multiple samples per action)
    """
    messages = []
    indices = []
    
    for sample in samples:
        condition_1 = sample["condition_1"]
        condition_2 = sample["condition_2"]
        
        # Select relevant conditions for info-gathering actions
        if "open" in sample["action"] or ("go to" in sample["action"] and 
            "drawer" not in sample["action"] and "cabinet" not in sample["action"] and
            "fridge" not in sample["action"] and "microwave" not in sample["action"]):
            target_recep = " ".join(sample["action"].split(" ")[-2:])
            # Randomly select conditions, prioritizing relevant ones
            condition_1_selected = random.sample(condition_1, min(config.world_model.max_condition_1_samples, len(condition_1)))
            for condition in condition_1:
                if target_recep in condition and condition not in condition_1_selected:
                    condition_1_selected.append(condition)
        else:
            condition_1_selected = []
        
        # Select relevant failed actions
        condition_2_selected = random.sample(condition_2, min(config.world_model.max_condition_2_samples, len(condition_2)))
        current_loc = sample["old_state"]["agent_location"]
        inventory = sample["old_state"]["inventory"]
        for condition in condition_2:
            if (current_loc in condition or inventory in condition or sample["action"] in condition) and condition not in condition_2_selected:
                condition_2_selected.append(condition)
        
        prompt = PREDICTION_PROMPT_TEMPLATE_METAICL.format(
            action=sample["action"],
            old_state=str(sample["old_state"]).replace("'", '"'),
            condition_1=condition_1_selected,
            condition_2=condition_2_selected
        )
        
        message = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": PREDICTION_SYSTEM_PROMPT
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_black}"}
                    }
                ]
            }
        ]
        
        # Use Monte Carlo sampling for info-gathering actions
        if sample["info_gather"]:
            indices.append((len(messages), len(messages) + config.world_model.num_mc_samples))
            messages.extend([message] * config.world_model.num_mc_samples)
        else:
            indices.append((len(messages), len(messages) + 1))
            messages.extend([message] * 1)
    
    # Get predictions in batches (prediction mode via [PRE] token in prompt)
    all_outputs = []
    batch_size = 250
    for i in tqdm(range(0, len(messages), batch_size), desc="Generating world model predictions"):
        outputs = get_vllm_response(
            messages=messages[i:i+batch_size],
            model=config.world_model.vllm_served_model_name,
            temperature=config.world_model.prediction_temperature,
            vllm_client=vllm_client,
            max_workers=100
        )
        all_outputs.extend(outputs)
    
    # Group outputs by sample
    outputs = []
    for index in indices:
        outputs.append([all_outputs[i] for i in range(index[0], index[1])])
    
    # Handle openable receptacles (predict objects inside)
    outputs = _predict_openable_receptacles(samples, outputs)
    
    return outputs


def _predict_openable_receptacles(
    samples: List[Dict[str, Any]],
    outputs: List[List[str]]
) -> List[List[str]]:
    """
    For "go to" actions targeting openable receptacles, predict objects inside.
    
    Args:
        samples: Original samples
        outputs: Current predictions
        
    Returns:
        Updated predictions with object lists for openable receptacles
    """
    indices_map = {}
    messages = []
    
    for i, sample in enumerate(samples):
        if "go to" in sample["action"] and sample["info_gather"] and (
            "drawer" in sample["action"] or "cabinet" in sample["action"] or
            "fridge" in sample["action"] or "microwave" in sample["action"] or
            "safe" in sample["action"]):
            
            state_changed = copy.deepcopy(sample["old_state"])
            target_recep = " ".join(sample["action"].split(" ")[-2:])
            state_changed["agent_location"] = target_recep
            
            # Select relevant conditions
            condition_1 = sample["condition_1"]
            condition_1_selected = random.sample(condition_1, min(config.world_model.max_condition_1_samples, len(condition_1)))
            for condition in condition_1:
                if target_recep in condition and condition not in condition_1_selected:
                    condition_1_selected.append(condition)
            
            prompt = PREDICTION_PROMPT_TEMPLATE_METAICL.format(
                action=sample["action"].replace("go to", "open"),
                old_state=str(state_changed).replace("'", '"'),
                condition_1=condition_1_selected,
                condition_2=[]
            )
            
            message = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": PREDICTION_SYSTEM_PROMPT
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_black}"}
                        }
                    ]
                }
            ]
            indices_map[i] = (len(messages), len(messages) + config.world_model.num_mc_samples)
            messages.extend([message] * config.world_model.num_mc_samples)
    
    # Get predictions for openable receptacles (same unified world model)
    all_outputs_changed = []
    batch_size = 250
    for i in tqdm(range(0, len(messages), batch_size), desc="Predicting openable receptacles"):
        outputs_ = get_vllm_response(
            messages=messages[i:i+batch_size],
            model=config.world_model.vllm_served_model_name,
            temperature=config.world_model.prediction_temperature,
            vllm_client=vllm_client,
            max_workers=100
        )
        all_outputs_changed.extend(outputs_)
    
    # Merge object predictions
    for i, index in indices_map.items():
        new_outputs = all_outputs_changed[index[0]:index[1]]
        for j, output in enumerate(new_outputs):
            try:
                objects = eval(output)["world_state"][" ".join(samples[i]["action"].split(" ")[-2:])]["objects"]
                outputs[i][j] = eval(outputs[i][j])
                outputs[i][j]["world_state"][" ".join(samples[i]["action"].split(" ")[-2:])]["objects"] = objects
            except Exception as e:
                print(f"Error merging openable receptacle prediction: {output}")
    
    return outputs
