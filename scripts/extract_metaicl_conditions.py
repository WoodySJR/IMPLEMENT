"""
Extract MetaICL Conditions from Evaluation Trajectories.
Extracts historical observations and failed actions for test-time adaptation.
"""

import os
import sys
import json
import hashlib
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def extract_conditions(eval_dir: str, output_dir: str):
    """
    Extract MetaICL conditions from evaluation trajectories.
    
    Args:
        eval_dir: Directory containing evaluation results
        output_dir: Output directory for condition files
    
    Returns:
        object_lists: Dictionary mapping task hashes to observed receptacles/objects
        failed_actions: Dictionary mapping task hashes to failed action tuples
    """
    os.makedirs(output_dir, exist_ok=True)
    
    object_lists = {}  # condition_1: historical observations
    failed_actions = {}  # condition_2: historical failed actions
    
    print("Extracting MetaICL conditions from evaluation trajectories...")
    
    for task_dir in os.listdir(eval_dir):
        if task_dir.endswith(".txt") or task_dir.endswith(".json") or task_dir.endswith(".png"):
            continue
        
        task_path = os.path.join(eval_dir, task_dir)
        if not os.path.isdir(task_path):
            continue
        
        for traj_dir in os.listdir(task_path):
            traj_path = os.path.join(task_path, traj_dir)
            if not os.path.isdir(traj_path):
                continue
            
            traj_file = os.path.join(traj_path, "traj.jsonl")
            if not os.path.exists(traj_file):
                continue
            
            # Generate hash key for this trajectory
            hash_key = hashlib.md5(f"{task_dir}_{traj_dir}".encode()).hexdigest()
            
            # Initialize containers
            if hash_key not in object_lists:
                object_lists[hash_key] = {}
            if hash_key not in failed_actions:
                failed_actions[hash_key] = []
            
            # Extract observations and failed actions
            with open(traj_file, "r") as f:
                for line in f:
                    try:
                        obs = json.loads(line)
                        state = obs.get("state", {})
                        world_state = state.get("world_state", {})
                        
                        # Extract visited receptacles and their objects (condition_1)
                        for receptacle in world_state:
                            if world_state[receptacle].get("visited", False) and receptacle not in object_lists[hash_key]:
                                # Store object types (without IDs)
                                object_lists[hash_key][receptacle] = [
                                    obj.split(" ")[0] for obj in world_state[receptacle].get("objects", [])
                                ]
                        
                        # Extract failed actions (condition_2)
                        action = obs.get("action_to_execute", obs.get("action", ""))
                        action_success = obs.get("action_success", True)
                        
                        # Skip invalid actions
                        if any(skip_word in action for skip_word in ["nothing", "examine", "help", "inventory", "look"]):
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
                            except Exception as e:
                                continue
                    
                    except Exception as e:
                        continue
    
    print(f"Extracted conditions for {len(object_lists)} trajectories")
    print(f"Average observations per trajectory: {sum(len(v) for v in object_lists.values()) / len(object_lists):.2f}")
    print(f"Average failed actions per trajectory: {sum(len(v) for v in failed_actions.values()) / len(failed_actions):.2f}")
    
    # Save conditions
    object_lists_file = os.path.join(output_dir, "object_lists.json")
    failed_actions_file = os.path.join(output_dir, "failed_actions.json")
    
    with open(object_lists_file, "w") as f:
        json.dump(object_lists, f, indent=2)
    print(f"Saved object lists to {object_lists_file}")
    
    with open(failed_actions_file, "w") as f:
        json.dump(failed_actions, f, indent=2)
    print(f"Saved failed actions to {failed_actions_file}")
    
    return object_lists, failed_actions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract MetaICL conditions")
    parser.add_argument("--eval_dir", type=str, required=True, help="Evaluation directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    
    args = parser.parse_args()
    
    extract_conditions(args.eval_dir, args.output_dir)

