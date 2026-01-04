"""
Data Collection Script for World Model Training.
Collects environment transitions using mixed behavioral policy.
"""

import os
import sys
import yaml
import json
import cv2
import random
import numpy as np
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alfworld.agents.environment.alfred_thor_env import AlfredThorEnv
from implement.utils.state_utils import obs_to_state, initial_obs_to_state


def collect_data(
    alfworld_root: str,
    output_dir: str,
    batch_size: int = 5,
    task_indices: list = None,
    prob_expert: float = 0.2,
    prob_admissible: float = 0.4,
    prob_inadmissible: float = 0.4
):
    """
    Collect training data from ALFWorld environment.
    
    Args:
        alfworld_root: Root directory of ALFWorld
        output_dir: Output directory for collected data
        batch_size: Number of parallel environments
        task_indices: Specific task indices to collect (None for all)
        prob_expert: Probability of expert actions
        prob_admissible: Probability of admissible actions
        prob_inadmissible: Probability of inadmissible actions
    """
    # Load config
    config_path = os.path.join(alfworld_root, "configs", "base_config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Configure environment
    config["env"]["thor"]["save_frames_to_disk"] = False
    config["controller"]["type"] = "oracle"
    config['env']['goal_desc_human_anns_prob'] = 0.0
    config['controller']['load_receps'] = True
    config['dataset']['num_train_games'] = -1
    config['env']['task_types'] = [1]  # Start with Pick & Place
    config["general"]["training_method"] = "dagger"
    config["dagger"]["training"]["max_nb_steps_per_episode"] = 30
    
    # Initialize environment
    all_admissible_commands = [[] for _ in range(batch_size)]
    env = AlfredThorEnv(config, train_eval="train", task_indices=task_indices)
    env.init_env(batch_size=batch_size)
    env.seed(928)
    
    obs, infos = env.reset()
    frames = env.get_frames()
    tasks = env.tasks
    goals = []
    dones = [False for _ in range(batch_size)]
    saves = [True for _ in range(batch_size)]
    states = [initial_obs_to_state(obs[i]) for i in range(batch_size)]
    
    step_num = 0
    task_dirs = []
    
    # Save initial state
    for i in range(batch_size):
        print(f"Task {i}: {tasks[i]}")
        
        # Create folder for each task
        task_dir = os.path.join(
            output_dir,
            tasks[i].split("train/")[1].split("/traj_data.json")[0]
        )
        os.makedirs(task_dir, exist_ok=True)
        task_dirs.append(task_dir)
        
        # Save initial frame
        cv2.imwrite(os.path.join(task_dir, "initial_frame.png"), frames[i])
        
        # Save initial state
        sample = {
            "initial_obs": obs[i],
            "goal": obs[i].split("Your task is to: ")[1],
            "frame_path": os.path.join(task_dir, "initial_frame.png"),
            "state": states[i]
        }
        goals.append(sample["goal"])
        
        with open(os.path.join(task_dir, "initial_state.jsonl"), "a") as f:
            f.write(json.dumps(sample) + "\n")
    
    # Accumulate admissible commands
    for i in range(batch_size):
        if not dones[i]:
            all_admissible_commands[i].extend(infos["admissible_commands"][i])
            all_admissible_commands[i] = list(set(all_admissible_commands[i]))
    
    first_step = True
    
    # Main interaction loop
    while not np.all(dones):
        actions = []
        
        if first_step:
            # Use admissible commands for first step
            for cmds in infos["admissible_commands"]:
                actions.append(random.choice(cmds))
            first_step = False
        else:
            # Mixed sampling strategy
            u = random.random()
            if u < prob_expert:
                print("Using expert actions")
                actions = infos["extra.expert_plan"]
                actions = [action[0] for action in actions]
            elif u < prob_expert + prob_admissible:
                print("Using admissible actions")
                actions = [random.choice(infos["admissible_commands"][i]) for i in range(batch_size)]
            else:
                print("Using inadmissible actions (for affordance learning)")
                actions = [random.choice(all_admissible_commands[i]) for i in range(batch_size)]
        
        # Execute actions
        actions = [action.replace("move", "put") for action in actions]
        obs, _, dones, infos = env.step(actions)
        print(f"Actions: {actions}")
        print(f"Observations: {obs}")
        
        # Update states
        new_states = []
        for i in range(batch_size):
            if saves[i]:
                new_states.append(obs_to_state(states[i], obs[i], actions[i]))
            else:
                new_states.append(None)
        
        # Update with other visible objects
        other_visible_objects = [infos["other_visible_objects"][i] for i in range(batch_size)]
        for i in range(batch_size):
            if len(other_visible_objects[i]) > 0 and saves[i]:
                for recep, objs in other_visible_objects[i].items():
                    if recep in states[i]["world_state"]:
                        if new_states[i]["world_state"][recep]["objects"] is None:
                            new_states[i]["world_state"][recep]["objects"] = objs
                        else:
                            new_states[i]["world_state"][recep]["objects"].extend(objs)
                            new_states[i]["world_state"][recep]["objects"] = list(set(new_states[i]["world_state"][recep]["objects"]))
        
        frames = env.get_frames()
        
        # Accumulate admissible commands
        for i in range(batch_size):
            if not dones[i]:
                all_admissible_commands[i].extend(infos["admissible_commands"][i])
                all_admissible_commands[i] = list(set(all_admissible_commands[i]))
        
        # Save transitions
        for i in range(batch_size):
            if saves[i]:
                task_dir = task_dirs[i]
                cv2.imwrite(os.path.join(task_dir, f"{step_num}.png"), frames[i])
                
                sample = {
                    "frame_path": os.path.join(task_dir, f"{step_num}.png"),
                    "state": new_states[i],
                    "text_obs": obs[i],
                    "action": actions[i],
                    "task": tasks[i],
                    "action_success": "Nothing happens" not in obs[i],
                    "other_visible_objects": other_visible_objects[i]
                }
                
                with open(os.path.join(task_dir, "traj.jsonl"), "a") as f:
                    f.write(json.dumps(sample) + "\n")
        
        states = new_states
        saves = [not dones[i] for i in range(batch_size)]
        step_num += 1
    
    env.close()
    print(f"Data collection complete. Saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect training data for world model")
    parser.add_argument("--alfworld_root", type=str, required=True, help="Root directory of ALFWorld")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size")
    parser.add_argument("--num_tasks", type=int, default=200, help="Number of tasks to collect")
    
    args = parser.parse_args()
    
    # Collect data in batches
    for i in range(0, args.num_tasks, args.batch_size):
        task_indices = list(range(i, min(i + args.batch_size, args.num_tasks)))
        try:
            collect_data(
                alfworld_root=args.alfworld_root,
                output_dir=args.output_dir,
                batch_size=args.batch_size,
                task_indices=task_indices
            )
            print(f"Successfully collected data for tasks {task_indices}")
        except Exception as e:
            print(f"Error collecting data for tasks {task_indices}: {e}")
            continue

