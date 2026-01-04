"""
Utilities for interacting with Language Models.
Handles API calls, parallel processing, and retry logic.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
from openai import OpenAI

from config import config


# Initialize OpenAI client
client = OpenAI(
    api_key=config.llm.api_key,
    base_url=config.llm.api_base_url
)


def get_policy_response(
    prompts: List[str],
    model: str,
    system_prompt: str,
    temperature: float = 1.0,
    max_retries: int = 3
) -> List[str]:
    """
    Get responses from policy LLM with parallel processing and retry logic.
    
    Args:
        prompts: List of prompt strings
        model: Model name
        system_prompt: System prompt
        temperature: Sampling temperature
        max_retries: Maximum number of retries on failure
        
    Returns:
        List of response strings
    """
    for attempt in range(max_retries):
        try:
            outputs = [None] * len(prompts)
            messages = [
                [
                    {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                    {"role": "user", "content": [{"type": "text", "text": prompt}]}
                ]
                for prompt in prompts
            ]
            
            with ThreadPoolExecutor(max_workers=config.llm.max_workers) as executor:
                futures = {
                    executor.submit(
                        client.chat.completions.create,
                        model=model,
                        messages=msg,
                        temperature=temperature
                    ): i
                    for i, msg in enumerate(messages)
                }
                for future in as_completed(futures):
                    outputs[futures[future]] = future.result().choices[0].message.content
            
            return outputs
            
        except Exception as e:
            print(f"Error in get_policy_response (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(100)
            else:
                raise


def get_belief_state_response(
    prompts: List[str],
    model: str,
    temperature: float = 1.0,
    max_retries: int = 3
) -> List[str]:
    """
    Get belief state summarization from LLM.
    
    Args:
        prompts: List of prompt strings
        model: Model name
        temperature: Sampling temperature
        max_retries: Maximum number of retries
        
    Returns:
        List of belief state strings
    """
    for attempt in range(max_retries):
        try:
            outputs = [None] * len(prompts)
            messages = [
                [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
                for prompt in prompts
            ]
            
            with ThreadPoolExecutor(max_workers=config.llm.max_workers) as executor:
                futures = {
                    executor.submit(
                        client.chat.completions.create,
                        model=model,
                        messages=msg,
                        temperature=temperature
                    ): i
                    for i, msg in enumerate(messages)
                }
                for future in as_completed(futures):
                    outputs[futures[future]] = future.result().choices[0].message.content
            
            return outputs
            
        except Exception as e:
            print(f"Error in get_belief_state_response (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(100)
            else:
                raise


def get_vllm_response(
    messages: List[List[Dict[str, Any]]],
    model: str,
    temperature: float,
    vllm_client: OpenAI,
    max_workers: int = 100
) -> List[str]:
    """
    Get responses from vLLM-deployed world model.
    
    Args:
        messages: List of message dictionaries
        model: Model name
        temperature: Sampling temperature
        vllm_client: vLLM OpenAI client
        max_workers: Maximum parallel workers
        
    Returns:
        List of response strings
    """
    outputs = [None] * len(messages)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                vllm_client.chat.completions.create,
                model=model,
                messages=message,
                temperature=temperature,
                top_p=0.9
            ): i
            for i, message in enumerate(messages)
        }
        for future in as_completed(futures):
            outputs[futures[future]] = future.result().choices[0].message.content
    
    return outputs
