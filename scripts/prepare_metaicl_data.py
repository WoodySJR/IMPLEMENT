"""
Prepare Training Data with MetaICL Conditions.
Augments training data with historical observations and failed actions.
"""

import os
import sys
import json
import hashlib
import random
import argparse
from tqdm import tqdm
import shutil
import copy

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from implement.prompts.world_model_prompts import (
    PERCEPTION_SYSTEM_PROMPT,
    PERCEPTION_PROMPT_TEMPLATE,
    PREDICTION_SYSTEM_PROMPT,
    PREDICTION_PROMPT_TEMPLATE_METAICL
)


def prepare_metaicl_data(
    data_dirs: list,
    output_dir: str
):
    """
    Prepare training data with MetaICL conditions.
    
    Args:
        data_dirs: List of directories containing collected data
        output_dir: Output directory for prepared data
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    
    # First pass: collect conditions
    print("Collecting MetaICL conditions...")
    object_lists = {}  # condition_1
    failed_actions = {}  # condition_2
    
    for data_dir in data_dirs:
        print(f"Processing {data_dir}")
        if not os.path.exists(data_dir):
            print(f"  Directory not found, skipping...")
            continue
        
        for task_dir in tqdm(os.listdir(data_dir)):
            if task_dir.endswith(".txt"):
                continue
            
            task_path = os.path.join(data_dir, task_dir)
            if not os.path.isdir(task_path):
                continue
            
            for traj_dir in os.listdir(task_path):
                traj_path = os.path.join(task_path, traj_dir)
                if not os.path.isdir(traj_path):
                    continue
                
                try:
                    hash_key = hashlib.md5(f"{task_dir}_{traj_dir}".encode()).hexdigest()
                    
                    # Initialize
                    if hash_key not in object_lists:
                        object_lists[hash_key] = {}
                    if hash_key not in failed_actions:
                        failed_actions[hash_key] = []
                    
                    # Read trajectory
                    traj_file = os.path.join(traj_path, "traj.jsonl")
                    if not os.path.exists(traj_file):
                        continue
                    
                    with open(traj_file, "r") as f:
                        for line in f:
                            obs = json.loads(line)
                            state = obs.get("state", {})
                            world_state = state.get("world_state", {})
                            
                            # Collect visited receptacles (condition_1)
                            for receptacle in world_state:
                                if world_state[receptacle].get("visited", False):
                                    if receptacle not in object_lists[hash_key]:
                                        object_lists[hash_key][receptacle] = [
                                            obj.split(" ")[0] for obj in world_state[receptacle].get("objects", [])
                                        ]
                            
                            # Collect failed actions (condition_2)
                            action = obs.get("action", "")
                            action_success = obs.get("action_success", True)
                            
                            # Skip invalid actions
                            if any(word in action for word in ["nothing", "examine", "help", "inventory", "look"]):
                                continue
                            if ("spatul" in action and "spatula" not in action):
                                continue
                            
                            if not action_success:
                                try:
                                    target_receptacle = " ".join(action.split(" ")[-2:])
                                    failed_tuple = (
                                        action,
                                        state.get("agent_location"),
                                        state.get("inventory"),
                                        world_state.get(target_receptacle, {})
                                    )
                                    if failed_tuple not in failed_actions[hash_key]:
                                        failed_actions[hash_key].append(failed_tuple)
                                except:
                                    continue
                
                except Exception as e:
                    continue
    
    print(f"Collected conditions for {len(object_lists)} trajectories")
    
    # Save conditions for later use
    conditions_dir = os.path.join(output_dir, "conditions")
    os.makedirs(conditions_dir, exist_ok=True)
    
    with open(os.path.join(conditions_dir, "object_lists.json"), "w") as f:
        json.dump(object_lists, f, indent=2)
    
    with open(os.path.join(conditions_dir, "failed_actions.json"), "w") as f:
        json.dump(failed_actions, f, indent=2)
    
    print(f"Saved MetaICL conditions to {conditions_dir}")
    
    # Second pass: create training samples with conditions
    print("Creating training samples with MetaICL conditions...")
    training_samples = []
    
    for data_dir in data_dirs:
        if not os.path.exists(data_dir):
            continue
        
        for task_dir in tqdm(os.listdir(data_dir)):
            if task_dir.endswith(".txt"):
                continue
            
            task_path = os.path.join(data_dir, task_dir)
            if not os.path.isdir(task_path):
                continue
            
            for traj_dir in os.listdir(task_path):
                traj_path = os.path.join(task_path, traj_dir)
                if not os.path.isdir(traj_path):
                    continue
                
                try:
                    hash_key = hashlib.md5(f"{task_dir}_{traj_dir}".encode()).hexdigest()
                    
                    # Load initial state
                    initial_state_file = os.path.join(traj_path, "initial_state.jsonl")
                    if not os.path.exists(initial_state_file):
                        continue
                    
                    with open(initial_state_file, "r") as f:
                        for line in f:
                            initial_state = json.loads(line)
                    
                    # Load trajectory
                    traj_file = os.path.join(traj_path, "traj.jsonl")
                    if not os.path.exists(traj_file):
                        continue
                    
                    traj_data = []
                    with open(traj_file, "r") as f:
                        for line in f:
                            traj_data.append(json.loads(line))
                    
                    # Process each transition
                    for i in range(len(traj_data)):
                        if i == 0:
                            old_state = initial_state["state"]
                            old_frame = "initial_frame.png"
                        else:
                            old_state = traj_data[i-1]["state"]
                            old_frame = f"{i-1}.png"
                        
                        action = traj_data[i]["action"]
                        new_state = traj_data[i]["state"]
                        new_frame = f"{i}.png"
                        
                        # Generate hash for images
                        old_frame_hash = hashlib.md5(f"{data_dir}_{task_dir}_{traj_dir}_{old_frame}".encode()).hexdigest()
                        new_frame_hash = hashlib.md5(f"{data_dir}_{task_dir}_{traj_dir}_{new_frame}".encode()).hexdigest()
                        
                        sample = {
                            "task_dir": task_dir,
                            "traj_dir": traj_dir,
                            "action": action,
                            "action_success": traj_data[i]["action_success"],
                            "old_state": old_state,
                            "new_state": new_state,
                            "old_frame_hash": old_frame_hash,
                            "new_frame_hash": new_frame_hash,
                            "other_visible_receptacles": list(traj_data[i].get("other_visible_objects", {}).keys())
                        }
                        
                        # Add MetaICL conditions for prediction samples
                        # Only add condition_1 for info-gathering actions (open or go to non-openable receptacles)
                        if "open" in action or ("go to" in action and 
                            "drawer" not in action and "cabinet" not in action and
                            "fridge" not in action and "microwave" not in action):
                            # Randomly sample number of conditions to include (0 to all)
                            if hash_key in object_lists and len(object_lists[hash_key]) > 0:
                                num_samples = random.choice(range(len(object_lists[hash_key]) + 1))
                                if num_samples > 0:
                                    sample["condition_1"] = random.sample(
                                        list(object_lists[hash_key].items()),
                                        num_samples
                                    )
                                else:
                                    sample["condition_1"] = []
                            else:
                                sample["condition_1"] = []
                        else:
                            sample["condition_1"] = []
                        
                        # Add condition_2 for all actions (randomly sample number of failed actions)
                        if hash_key in failed_actions and len(failed_actions[hash_key]) > 0:
                            num_samples = random.choice(range(len(failed_actions[hash_key]) + 1))
                            if num_samples > 0:
                                sample["condition_2"] = random.sample(
                                    failed_actions[hash_key],
                                    num_samples
                                )
                            else:
                                sample["condition_2"] = []
                        else:
                            sample["condition_2"] = []
                        
                        training_samples.append(sample)
                        
                        # Copy images
                        old_frame_path = os.path.join(traj_path, old_frame)
                        new_frame_path = os.path.join(traj_path, new_frame)
                        
                        if os.path.exists(old_frame_path):
                            shutil.copy(
                                old_frame_path,
                                os.path.join(output_dir, "images", f"{old_frame_hash}.png")
                            )
                        if os.path.exists(new_frame_path):
                            shutil.copy(
                                new_frame_path,
                                os.path.join(output_dir, "images", f"{new_frame_hash}.png")
                            )
                
                except Exception as e:
                    continue
    
    # Save training samples
    print(f"Total training samples with MetaICL: {len(training_samples)}")
    
    with open(os.path.join(output_dir, "training_samples_metaicl.jsonl"), "w") as f:
        for sample in training_samples:
            f.write(json.dumps(sample) + "\n")
    
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract MetaICL conditions")
    parser.add_argument("--eval_dir", type=str, required=True, help="Evaluation directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    
    args = parser.parse_args()
    
    extract_conditions(args.eval_dir, args.output_dir)

