import os

from dataclasses import dataclass
from .helpers import get_default_device, get_most_recent_version

@dataclass
class BaseConfig:
    DEVICE = get_default_device()
    DATASET = "Cifar-10" #  "MNIST", "Cifar-10", "Cifar-100", "Flowers"
    
    working_dir = os.getcwd()
    # For logging inferece images and saving checkpoints.
    root_log_dir = os.path.join("Log", "Inference")
    root_checkpoint_dir = os.path.join("Log", "checkpoints")

    # Current log and checkpoint directory.
    recent_version = get_most_recent_version(root_checkpoint_dir)
    log_dir = os.path.join(root_log_dir, recent_version)
    checkpoint_dir = os.path.join(root_checkpoint_dir, recent_version)

@dataclass
class TrainingConfig:
    TIMESTEPS = 1000 # Define number of diffusion timesteps
    IMG_SHAPE = (1, 32, 32) if BaseConfig.DATASET == "MNIST" else (3, 32, 32) 
    NUM_EPOCHS = 30
    BATCH_SIZE = 128
    LR = 2e-4
    NUM_WORKERS = 2