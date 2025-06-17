import torch

# --- General Settings ---
# Set the device to use for training
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Set the number of output classes for your model
NUM_CLASSES = 10  # Example: 10 for CIFAR-10, 3 for Food-Vision Mini
# Set a fixed batch size for all experiments
BATCH_SIZE = 32
# Set a fixed learning rate for all experiments
LEARNING_RATE = 0.001


# --- Data & Model Paths ---
# Define the name for the dataset you are using (for logging purposes)
DATA_NAME = "FakeData_Example"
# Directory where trained models will be saved
MODEL_SAVE_DIR = "models"
# Directory where TensorBoard logs will be saved
LOG_DIR = "runs"


# --- Experiment Grid ---
# Define all the experiments you want to run.
# Each dictionary in the list represents one full experiment.
#
# "model_name" must be a key in the SUPPORTED_MODELS dictionary in model_builder.py
# "epochs" is the number of epochs to train for.
#
EXPERIMENTS = [
    {
        "model_name": "effnetb0",
        "epochs": 5,
    },
    {
        "model_name": "effnetb2",
        "epochs": 5,
    },
    {
        "model_name": "effnetb0",
        "epochs": 10,
    },
    {
        "model_name": "effnetb2",
        "epochs": 10,
    },
    # --- ADD YOUR NEW EXPERIMENTS HERE ---
    # Example: Running a ResNet18 model
    # {
    #    "model_name": "resnet18",
    #    "epochs": 15,
    # },
]
