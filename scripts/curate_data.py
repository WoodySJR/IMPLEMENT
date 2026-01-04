"""
Data Curation Script.
Formats collected samples into final training format with prompts.
Supports both basic and MetaICL versions.
"""

import os
import sys
import json
import argparse
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from implement.prompts.world_model_prompts import (
    PERCEPTION_SYSTEM_PROMPT,
    PERCEPTION_PROMPT_TEMPLATE,
    PREDICTION_SYSTEM_PROMPT,
    PREDICTION_PROMPT_TEMPLATE_METAICL
)


def curate_data(
    input_file: str,
    data_dir: str,
    output_dir: str,
    black_image_path: str = None
):
    """
    Curate training samples into final format with prompts.
    
    Args:
        input_file: Path to training_samples.jsonl (with or without MetaICL conditions)
        data_dir: Directory containing images
        output_dir: Output directory
        black_image_path: Path to black placeholder image for prediction
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Use black image from assets if not provided
    if black_image_path is None:
        black_image_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "assets", "black.png"
        )
    
    # Load samples
    samples = []
    print(f"Loading samples from {input_file}")
    with open(input_file, "r") as f:
        for line in f:
            sample = json.loads(line)
            samples.append(sample)
    
    print(f"Total samples: {len(samples)}")
    
    curated_samples_perception = []
    curated_samples_prediction = []
    
    # Format each sample
    print("Formatting samples...")
    for sample in tqdm(samples):
        # Perception sample
        perception_prompt = PERCEPTION_PROMPT_TEMPLATE.format(
            action=sample["action"],
            action_success=sample["action_success"],
            old_state=str(sample["old_state"]).replace("'", '"'),
            other_visible_receptacles=sample["other_visible_receptacles"]
        )
        
        curated_sample_perception = [
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
                        "text": perception_prompt
                    },
                    {
                        "type": "image",
                        "image": os.path.join(data_dir, "images", f"{sample['new_frame_hash']}.png")
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": str(sample["new_state_perception"]).replace("'", '"')
                    }
                ]
            }
        ]
        curated_samples_perception.append(curated_sample_perception)
        
        # Prediction sample (with MetaICL if conditions are present)
        prediction_prompt = PREDICTION_PROMPT_TEMPLATE_METAICL.format(
            action=sample["action"],
            old_state=str(sample["old_state"]).replace("'", '"'),
            condition_1=sample.get("condition_1", []),  # Empty list if not present
            condition_2=sample.get("condition_2", [])   # Empty list if not present
        )
        
        curated_sample_prediction = [
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
                        "text": prediction_prompt
                    },
                    {
                        "type": "image",
                        "image": black_image_path  # Placeholder black image
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": str(sample["new_state_prediction"]).replace("'", '"')
                    }
                ]
            }
        ]
        curated_samples_prediction.append(curated_sample_prediction)
    
    # Save curated samples
    print("Saving curated samples...")
    perception_file = os.path.join(output_dir, "curated_samples_perception.jsonl")
    with open(perception_file, "w") as f:
        for sample in curated_samples_perception:
            f.write(json.dumps(sample) + "\n")
    print(f"Saved {len(curated_samples_perception)} perception samples to {perception_file}")
    
    prediction_file = os.path.join(output_dir, "curated_samples_prediction.jsonl")
    with open(prediction_file, "w") as f:
        for sample in curated_samples_prediction:
            f.write(json.dumps(sample) + "\n")
    print(f"Saved {len(curated_samples_prediction)} prediction samples to {prediction_file}")
    
    print("Curation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Curate training data")
    parser.add_argument("--input_file", type=str, required=True, 
                        help="Path to training_samples.jsonl")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing images")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--black_image", type=str, default=None,
                        help="Path to black placeholder image")
    
    args = parser.parse_args()
    
    curate_data(
        input_file=args.input_file,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        black_image_path=args.black_image
    )
