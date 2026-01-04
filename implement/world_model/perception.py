"""
World Model Perception Module.
Handles visual perception and state extraction from images.
"""

import base64
import time
from typing import List, Dict, Any
from openai import OpenAI
from tqdm import tqdm

from config import config
from implement.prompts.world_model_prompts import (
    PERCEPTION_SYSTEM_PROMPT,
    PERCEPTION_PROMPT_TEMPLATE
)
from implement.utils.llm_utils import get_vllm_response


# Initialize vLLM client for perception
vllm_client = OpenAI(
    base_url=f"http://{config.world_model.vllm_host}:{config.world_model.vllm_port}/v1",
    api_key=config.world_model.vllm_api_key
)


def world_model_perception(samples: List[Dict[str, Any]]) -> List[str]:
    """
    Perform perception using the world model.
    Extracts symbolic state from visual observations.
    
    Args:
        samples: List of sample dictionaries, each containing:
            - action: Action that was executed
            - action_success: Whether action was successful
            - old_state: State before action
            - new_frame: Base64-encoded image after action
            - other_visible_receptacles: List of other visible receptacles
            
    Returns:
        List of predicted state strings (JSON format)
    """
    print("Generating world model perceptions...")
    start_time = time.time()
    
    messages = []
    for sample in samples:
        prompt = PERCEPTION_PROMPT_TEMPLATE.format(
            action=sample["action"],
            action_success=sample["action_success"],
            old_state=str(sample["old_state"]).replace("'", '"'),
            other_visible_receptacles=sample["other_visible_receptacles"]
        )
        
        message = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": PERCEPTION_SYSTEM_PROMPT
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
                        "image_url": {"url": f"data:image/jpeg;base64,{sample['new_frame']}"}
                    }
                ]
            }
        ]
        messages.append(message)
    
    # Get responses from unified world model (perception mode via [PER] token in prompt)
    outputs = get_vllm_response(
        messages=messages,
        model=config.world_model.vllm_served_model_name,
        temperature=0,  # Deterministic for perception
        vllm_client=vllm_client,
        max_workers=100
    )
    
    end_time = time.time()
    print(f"Perception time: {end_time - start_time:.2f} seconds")
    
    return outputs
