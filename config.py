"""
IMPLEMENT Configuration File
Contains all hyperparameters and settings for the framework.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class LLMConfig:
    """Configuration for Language Models"""
    # API settings (replace with your own)
    api_base_url: str = "YOUR_API_BASE_URL"
    api_key: str = "YOUR_API_KEY"
    
    # Model names
    policy_model: str = "gpt-4.1-mini"  # Options: gpt-4.1, gpt-4.1-mini, gemini-2.5-flash, qwen2.5-vl-72b-instruct
    belief_state_model: str = "gpt-4.1-mini"
    
    # Generation parameters
    temperature: float = 1.0
    max_workers: int = 20


@dataclass
class WorldModelConfig:
    """Configuration for Unified World Model (handles both perception and prediction)"""
    # Model settings
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    model_cache_dir: str = "./hf-models"
    
    # vLLM deployment (single unified model)
    vllm_host: str = "localhost"
    vllm_port: int = 8002
    vllm_api_key: str = "sk-xxx"
    vllm_served_model_name: str = "wm_metaicl"  # Must match --served-model-name in vLLM
    
    # vLLM performance settings
    tensor_parallel_size: int = 4
    gpu_memory_utilization: float = 0.85
    enable_prefix_caching: bool = True
    max_num_batched_tokens: int = 8192
    
    # Prediction settings
    prediction_temperature: float = 1.0  # Temperature for Monte Carlo rollouts
    num_mc_samples: int = 10  # Number of Monte Carlo samples for info-gathering actions
    
    # MetaICL settings
    max_condition_1_samples: int = 15  # Max historical observations to include
    max_condition_2_samples: int = 15  # Max failed actions to include


@dataclass
class PolicyConfig:
    """Configuration for Policy Iteration"""
    max_interaction_rounds: int = 5  # Maximum LLM-WM interaction rounds per step
    max_episode_steps: int = 30  # Maximum steps per episode
    batch_size: int = 5  # Number of parallel environments


@dataclass
class TrainingConfig:
    """Configuration for World Model Training"""
    # Data settings
    data_dir: str = "./data/world_model_training"
    output_dir: str = "./checkpoints"
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    
    # Training hyperparameters (from paper)
    num_epochs: int = 3
    learning_rate: float = 2e-5
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 0.2
    
    # LoRA settings (from paper: r=16, α=16)
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: str = "all-linear"
    
    # Optimizer settings
    optimizer: str = "adamw_torch_fused"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    weight_decay: float = 0.01
    
    # Validation
    val_split: float = 0.01
    
    # Logging (optional)
    wandb_project: str = "implement-world-model"
    wandb_run_name: str = "training-run"


@dataclass
class EnvironmentConfig:
    """Configuration for ALFWorld Environment"""
    alfworld_config_path: str = "./configs/base_config.yaml"
    task_types: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6])
    # Task types: 1-Pick&Place, 2-Examine in Light, 3-Clean&Place, 
    #             4-Heat&Place, 5-Cool&Place, 6-Pick Two&Place
    
    seed: int = 928
    save_frames: bool = False


@dataclass
class APIConfig:
    """Configuration for FastAPI Server"""
    host: str = "localhost"
    port: int = 30972


@dataclass
class Config:
    """Main configuration class"""
    llm: LLMConfig = field(default_factory=LLMConfig)
    world_model: WorldModelConfig = field(default_factory=WorldModelConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    api: APIConfig = field(default_factory=APIConfig)


# Global config instance
config = Config()


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from file or use defaults.
    
    Args:
        config_path: Path to config file (YAML or JSON). If None, use defaults.
    
    Returns:
        Config object
    """
    if config_path is None:
        return Config()
    
    # TODO: Implement loading from YAML/JSON file
    import yaml
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Update config with loaded values
    # This is a simplified version - you may want to add more sophisticated merging
    return Config(**config_dict)

