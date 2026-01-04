"""
Evaluation Script for IMPLEMENT.
Evaluates the framework on ALFWorld tasks.
"""

import os
import sys
import yaml
import json
import cv2
import base64
import argparse
import requests
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alfworld.agents.environment.alfred_thor_env import AlfredThorEnv
from implement.utils.state_utils import initial_obs_to_state, obs_to_state


def evaluate(
    alfworld_root: str,
    api_url: str,
    model: str,
    output_dir: str,
    batch_size: int = 5,
    max_trials: int = 12
):
    """
    Evaluate IMPLEMENT on ALFWorld.
    
    Args:
        alfworld_root: Root directory of ALFWorld
        api_url: URL of IMPLEMENT API server
        model: Policy model name
        output_dir: Output directory for results
        batch_size: Number of parallel environments
        max_trials: Maximum number of trials per task
    """
    # Load config
    config_path = os.path.join(alfworld_root, "configs", "base_config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Configure environment
    config["env"]["thor"]["save_frames_to_disk"] = False
    config["controller"]["type"] = "oracle"
    config['env']['goal_desc_human_anns_prob'] = 0.0
    config['controller']['load_receps'] = False
    config['dataset']['num_eval_games'] = -1
    config['env']['task_types'] = [1, 2, 3, 4, 5, 6]
    config["general"]["training_method"] = "dagger"
    config["dagger"]["training"]["max_nb_steps_per_episode"] = 30
    
    # Initialize environment
    env = AlfredThorEnv(config, train_eval="eval_out_of_distribution")
    env.init_env(batch_size=batch_size)
    env.seed(928)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Track success across trials
    all_tasks = []
    success_counts = {}
    
    for trial in range(max_trials):
        print(f"\n=== Trial {trial + 1}/{max_trials} ===")
        
        obs, infos = env.reset()
        frames = env.get_frames()
        tasks = env.tasks
        
        dones = [False] * batch_size
        states = [initial_obs_to_state(obs[i]) for i in range(batch_size)]
        histories = [[] for _ in range(batch_size)]
        
        # Initialize conditions for MetaICL (empty for first trial)
        conditions_1 = [[] for _ in range(batch_size)]
        conditions_2 = [[] for _ in range(batch_size)]
        
        step_num = 0
        max_steps = 30
        
        while not np.all(dones) and step_num < max_steps:
            # Prepare API request
            current_goals = [obs[i].split("Your task is to: ")[1] for i in range(batch_size) if not dones[i]]
            current_states = [states[i] for i in range(batch_size) if not dones[i]]
            current_histories = [histories[i] for i in range(batch_size) if not dones[i]]
            current_tasks = [tasks[i] for i in range(batch_size) if not dones[i]]
            current_conditions_1 = [conditions_1[i] for i in range(batch_size) if not dones[i]]
            current_conditions_2 = [conditions_2[i] for i in range(batch_size) if not dones[i]]
            
            # Get actions from API
            try:
                response = requests.post(
                    f"{api_url}/generate_policy",
                    json={
                        "current_goals": current_goals,
                        "current_states": current_states,
                        "histories": current_histories,
                        "current_tasks": current_tasks,
                        "model": model,
                        "conditions_1": current_conditions_1,
                        "conditions_2": current_conditions_2
                    },
                    timeout=300
                )
                actions = response.json()
            except Exception as e:
                print(f"Error calling API: {e}")
                break
            
            # Execute actions
            actions_to_execute = []
            for i, done in enumerate(dones):
                if not done:
                    actions_to_execute.append(actions.pop(0))
                else:
                    actions_to_execute.append("")
            
            obs, _, dones, infos = env.step(actions_to_execute)
            frames = env.get_frames()
            
            # Update states via perception
            new_states = []
            for i in range(batch_size):
                if not dones[i]:
                    # Encode frame
                    _, buffer = cv2.imencode('.png', frames[i])
                    b64_img = base64.b64encode(buffer).decode('utf-8')
                    
                    # Call perception API
                    try:
                        response = requests.post(
                            f"{api_url}/perception",
                            json={
                                "samples": [{
                                    "action": actions_to_execute[i],
                                    "action_success": "Nothing happens" not in obs[i],
                                    "old_state": states[i],
                                    "new_frame": b64_img,
                                    "other_visible_receptacles": list(infos["other_visible_objects"][i].keys())
                                }]
                            },
                            timeout=60
                        )
                        new_state = eval(response.json()[0].replace("'", '"'))
                        new_states.append(new_state)
                    except Exception as e:
                        print(f"Error in perception: {e}")
                        new_states.append(states[i])
                else:
                    new_states.append(None)
            
            # Update histories
            for i in range(batch_size):
                if not dones[i]:
                    if "Nothing happens" not in obs[i]:
                        histories[i].append(f"{len(histories[i])+1}. You successfully {actions_to_execute[i]}.")
                    else:
                        histories[i].append(f"{len(histories[i])+1}. You attempt to {actions_to_execute[i]} but fail.")
            
            states = new_states
            step_num += 1
        
        # Record results
        for i, task in enumerate(tasks):
            if task not in all_tasks:
                all_tasks.append(task)
                success_counts[task] = 0
            
            if infos["won"][i]:
                success_counts[task] += 1
        
        # Print current success rate
        num_success = sum(1 for count in success_counts.values() if count > 0)
        print(f"Success rate @ trial {trial + 1}: {num_success}/{len(all_tasks)} = {num_success/len(all_tasks)*100:.1f}%")
    
    # Save final results
    results = {
        "model": model,
        "max_trials": max_trials,
        "total_tasks": len(all_tasks),
        "success_counts": success_counts,
        "success_rate": sum(1 for count in success_counts.values() if count > 0) / len(all_tasks)
    }
    
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n=== Final Results ===")
    print(f"Success rate: {results['success_rate']*100:.1f}%")
    print(f"Results saved to {output_dir}/results.json")
    
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate IMPLEMENT")
    parser.add_argument("--alfworld_root", type=str, required=True, help="ALFWorld root directory")
    parser.add_argument("--api_url", type=str, default="http://localhost:30972", help="API URL")
    parser.add_argument("--model", type=str, default="gpt-4.1-mini", help="Policy model")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size")
    parser.add_argument("--max_trials", type=int, default=12, help="Max trials per task")
    
    args = parser.parse_args()
    
    evaluate(
        alfworld_root=args.alfworld_root,
        api_url=args.api_url,
        model=args.model,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_trials=args.max_trials
    )

