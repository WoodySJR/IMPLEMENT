"""
World Model Training Script.
Trains the unified world model using collected transition data with MetaICL.
"""

import os
import sys
import torch
import json
import random
import argparse
from functools import partial
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, TrainingArguments
from trl import SFTTrainer
from peft import get_peft_model, LoraConfig
from qwen_vl_utils import process_vision_info

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config


def find_assistant_content_indexes(token_ids):
    """
    Find the indexes of assistant response in tokenized sequence.
    Used for label masking during training.
    """
    start_indexes = []
    end_indexes = []
    
    # <|im_start|>assistant\n = [151644, 77091, 198]
    # <|im_end|>\n = [151645, 198]
    
    for i in range(len(token_ids) - 2):
        if token_ids[i] == 151644 and token_ids[i+1] == 77091 and token_ids[i+2] == 198:
            start_indexes.append(i + 3)
            # Find corresponding end token
            for j in range(i + 3, len(token_ids) - 1):
                if token_ids[j] == 151645 and token_ids[j+1] == 198:
                    end_indexes.append(j + 2)  # Include <|im_end|>\n tokens
                    break
    
    return list(zip(start_indexes, end_indexes))


def collate_fn(examples, processor):
    """
    Collate function for training with label masking.
    Only computes loss on assistant responses.
    """
    # Apply chat template
    texts = [processor.apply_chat_template(example, tokenize=False) for example in examples]
    image_inputs = [process_vision_info(example)[0] for example in examples]
    
    # Tokenize
    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
    
    # Create labels with masking
    input_ids_list = batch["input_ids"].tolist()
    labels_list = []
    
    for ids_list in input_ids_list:
        label_ids = [-100] * len(ids_list)  # Mask everything by default
        # Unmask only assistant responses
        for begin_idx, end_idx in find_assistant_content_indexes(ids_list):
            label_ids[begin_idx:end_idx] = ids_list[begin_idx:end_idx]
        labels_list.append(label_ids)
    
    batch["labels"] = torch.tensor(labels_list, dtype=torch.int64)
    
    return batch


def train_world_model(
    data_dir: str,
    output_dir: str,
    model_name: str = None,
    num_epochs: int = None,
    use_wandb: bool = True
):
    """
    Train the unified world model with MetaICL.
    
    Args:
        data_dir: Directory containing curated training data
        output_dir: Directory to save checkpoints
        model_name: Model name (default from config)
        num_epochs: Number of epochs (default from config)
        use_wandb: Whether to use Weights & Biases logging
    """
    model_name = model_name or config.training.model_name
    num_epochs = num_epochs or config.training.num_epochs
    
    print(f"Training unified world model: {model_name}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load processor
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(
        model_name,
        cache_dir=config.world_model.model_cache_dir
    )
    
    # Define collate function
    collate_fn_for_trainer = partial(collate_fn, processor=processor)
    
    # Load model
    print("Loading model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        device_map="balanced",  # Automatically balance across GPUs
        torch_dtype=torch.bfloat16,
        cache_dir=config.world_model.model_cache_dir
    )
    
    # Initialize logging
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project=config.training.wandb_project,
                name=config.training.wandb_run_name,
                config={
                    "model": model_name,
                    "epochs": num_epochs,
                    "learning_rate": config.training.learning_rate,
                    "batch_size": config.training.per_device_train_batch_size,
                    "lora_r": config.training.lora_r,
                }
            )
        except ImportError:
            print("wandb not available, skipping logging")
            use_wandb = False
    
    # Configure LoRA
    print("Configuring LoRA...")
    peft_config = LoraConfig(
        r=config.training.lora_r,
        lora_alpha=config.training.lora_alpha,
        lora_dropout=config.training.lora_dropout,
        target_modules=config.training.lora_target_modules,
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Load dataset
    print("Loading dataset...")
    dataset = []
    
    # Load perception samples
    perception_file = os.path.join(data_dir, "curated_samples_perception.jsonl")
    if os.path.exists(perception_file):
        with open(perception_file, "r") as f:
            for line in f:
                sample = json.loads(line)
                dataset.append(sample)
        print(f"Loaded {len(dataset)} perception samples")
    
    # Load prediction samples
    prediction_file = os.path.join(data_dir, "curated_samples_prediction.jsonl")
    if os.path.exists(prediction_file):
        perception_count = len(dataset)
        with open(prediction_file, "r") as f:
            for line in f:
                sample = json.loads(line)
                dataset.append(sample)
        print(f"Loaded {len(dataset) - perception_count} prediction samples")
    
    print(f"Total samples: {len(dataset)}")
    
    # Split dataset
    val_size = int(len(dataset) * config.training.val_split)
    val_dataset = random.sample(dataset, val_size)
    train_dataset = [sample for sample in dataset if sample not in val_dataset]
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        max_grad_norm=config.training.max_grad_norm,
        optim=config.training.optimizer,
        adam_beta1=config.training.adam_beta1,
        adam_beta2=config.training.adam_beta2,
        weight_decay=config.training.weight_decay,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        bf16=True,
        report_to="wandb" if use_wandb else "none",
        remove_unused_columns=False,
    )
    
    # Initialize trainer
    print("Initializing trainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn_for_trainer,
        processing_class=processor.tokenizer,  # Use processing_class instead of tokenizer
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save final model
    print(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    
    if use_wandb:
        try:
            import wandb
            wandb.finish()
        except:
            pass
    
    print("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train unified world model with MetaICL")
    parser.add_argument("--data_dir", type=str, required=True, help="Data directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--model_name", type=str, default=None, help="Model name")
    parser.add_argument("--num_epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--no_wandb", action="store_true", help="Disable Weights & Biases")
    
    args = parser.parse_args()
    
    train_world_model(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        num_epochs=args.num_epochs,
        use_wandb=not args.no_wandb
    )
