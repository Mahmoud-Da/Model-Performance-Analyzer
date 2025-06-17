# PyTorch Experiment Tracker

A flexible and reusable tool for running and tracking multiple PyTorch experiments. This framework is designed to let you define a grid of experiments (e.g., trying different models or numbers of epochs) in a single configuration file and then automatically run, log, and save the results for each one.

## Features

- **Configuration-Driven**: Define all your experiments in a single `config.py` file. No need to modify the core logic.
- **Extensible Model Support**: Easily add new models from `torchvision` (or your custom models) by making a one-line change.
- **Automated Logging**: Automatically logs training/testing loss and accuracy to TensorBoard for each experiment.
- **Organized Outputs**: Saves TensorBoard logs and trained model (`.pth`) files into structured directories.
- **Modular Code**: The logic is separated into an engine, model builder, data setup, and utilities for clarity and maintenance.

## Project Structure

```
pytorch_experiment_tracker/
├── main.py                 # Main executable script to run experiments
├── config.py               # <-- THE ONLY FILE YOU NEED TO MODIFY
├── engine.py               # Core training and testing logic
├── model_builder.py        # Factory for creating models (add new models here)
├── data_setup.py           # Data loading logic
├── utils.py                # Helper functions (e.g., saving models)
├── Pipfile                 # Project dependencies
└── README.md               # This file
```

## How to Use

### Step 1: Setup

1.  **Clone the repository:**

    ```bash
    git clone <your-repo-url>
    cd Model-Performance-Analyzer
    ```

2.  **Build the Image**

    ```bash
    docker compose build
    ```

3.  **Run the container**
    ```bash
    docker compose up
    ```

### Step 2: Configure Your Experiments

Open the `config.py` file. This is your control panel.

1.  **General Settings:** Adjust `DEVICE`, `NUM_CLASSES`, `BATCH_SIZE`, and `LEARNING_RATE` for your project.
2.  **Define Experiments:** Modify the `EXPERIMENTS` list. Each dictionary in this list defines one experiment. You can add as many as you want.

**Example `config.py`:**

```python
# To run experiments on EffNetB0 and ResNet18 for 5 and 10 epochs:
EXPERIMENTS = [
    {
        "model_name": "effnetb0",
        "epochs": 5,
    },
    {
        "model_name": "resnet18",
        "epochs": 5,
    },
    {
        "model_name": "effnetb0",
        "epochs": 10,
    },
    {
        "model_name": "resnet18",
        "epochs": 10,
    },
]
```

### Step 3: Add a New Model (Optional)

If you want to use a model that isn't already listed (e.g., `densenet121`), you only need to make a small change in `model_builder.py`.

1.  Open `model_builder.py`.
2.  Add a new entry to the `SUPPORTED_MODELS` dictionary.

**Example: Adding `DenseNet121`**
Find the model in `torchvision.models`, its default weights, and the number of `in_features` for its classifier. Then add it to the dictionary:

```python
# In model_builder.py

SUPPORTED_MODELS = {
    # ... existing models
    "resnet50": (torchvision.models.resnet50, torchvision.models.ResNet50_Weights.DEFAULT, 2048),
    # Add this new line:
    "densenet121": (torchvision.models.densenet121, torchvision.models.DenseNet121_Weights.DEFAULT, 1024),
}
```

Now you can use `"densenet121"` as a `model_name` in your `config.py`.

### Step 4: Run the Experiments

Execute the main script from your terminal. It will automatically read your `config.py` and run all defined experiments sequentially.

```bash
python3 main.py
```

### Step 5: Analyze the Results

The script generates two key outputs:

- **Saved Models**: In the `models/` directory.
- **TensorBoard Logs**: In the `runs/` directory, organized by date, data, model, and epoch count.

To visualize the training progress and compare experiments, launch TensorBoard:

```bash
tensorboard --logdir runs
```

Open the URL provided by TensorBoard (usually `http://localhost:6006`) in your web browser. You can now compare the performance curves of all your experiments on interactive charts.
