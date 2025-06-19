# PyTorch Performance Analyzer


A flexible and reusable tool for running and tracking multiple PyTorch experiments. This framework is designed to let you define a grid of experiments (e.g., trying different models or numbers of epochs) in a single configuration file and then automatically run, log, and save the results for each one.

## Features

- **Configuration-Driven**: Define all your experiments in a single `config.py` file. No need to modify the core logic.
- **Extensible Model Support**: Easily add new models from `torchvision` (or your custom models) by making a one-line change.
- **Automated Logging**: Automatically logs training/testing loss and accuracy to TensorBoard for each experiment.
- **Organized Outputs**: Saves TensorBoard logs and trained model (`.pth`) files into structured directories.
- **Modular Code**: The logic is separated into an engine, model builder, data setup, and utilities for clarity and maintenance.

## Project Structure

```
model_performance_analyzer/
├── main.py                 # Main executable script to run experiments
├── config.py               # <-- THE ONLY FILE YOU NEED TO MODIFY
├── engine.py               # Core training and testing logic
├── model_builder.py        # Factory for creating models (add new models here)
├── data_setup.py           # Data loading logic
├── utils.py                # Helper functions (e.g., saving models)
├── Pipfile                 # Project dependencies
└── README.md               # This file
```

## Example Results
![Screenshot 2025-06-18 at 5 19 38](https://github.com/user-attachments/assets/1a6a2f63-033e-4ae0-b52f-16450b244e4a)



![Screenshot 2025-06-18 at 5 20 22](https://github.com/user-attachments/assets/f2dc3871-7e1b-4c06-a8bc-0b3b4c459dad)

## How to Use

### Step 1: Setup

1.  **Clone the repository:**

    ```bash
    git clone <repo-url>
    cd Model-Performance-Analyzer
    ```

2.  **Build the Image**

    ```bash
    docker compose build
    ```

3.  **Run the container**

    ```bash
    docker compose up -d
    ```

4.  **Access the container**

    ```bash
    docker compose exec app bash
    ```

5.  **Run the main script**

    ```bash
    python3 main.py
    ```

6.  **view the experiment**
    ```bash
    tensorboard --logdir runs --bind_all
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

# Running Your PyTorch Project with Docker

This document outlines the steps to build and run this PyTorch application using Docker and Docker Compose. This ensures a consistent and reproducible environment for development and deployment.

## Prerequisites

1.  **Docker**: Ensure Docker Desktop (for Mac/Windows) or Docker Engine (for Linux) is installed and running. You can download it from [docker.com](https://www.docker.com/products/docker-desktop/).
2.  **Docker Compose**: Docker Compose V2 is typically included with Docker Desktop. For Linux, you might need to install it separately.
3.  **(Optional) NVIDIA GPU Support**:
    - If you intend to use NVIDIA GPUs, ensure you have the latest NVIDIA drivers installed on your host machine.
    - Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) on your host machine. This allows Docker containers to access NVIDIA GPUs.
4.  **Project Files**:
    - `Dockerfile`: Defines the Docker image for the application.
    - `docker-compose.yml`: Defines how to run the application services (including GPU support).
    - `Pipfile`: Specifies Python package dependencies.
    - `Pipfile.lock`: Locks package versions for reproducible builds.
    - Your application code (e.g., `inference.py`).

## Building and Running the Application

We will use Docker Compose to manage the build and run process.

### Step 1: Clone the Repository (if applicable)

If you haven't already, clone the project repository to your local machine:

```bash
git clone <your-repository-url>
cd <your-project-directory>
```

### Step 2: Check/Generate Pipfile.lock

The `Dockerfile` uses `pipenv install --deploy`, which requires `Pipfile.lock` to be up-to-date with `Pipfile`.

**Troubleshooting `Pipfile.lock` out-of-date error:**
If, during the Docker build process (Step 3), you encounter an error similar to:

```
Your Pipfile.lock (...) is out of date. Expected: (...).
ERROR:: Aborting deploy
```

This means your `Pipfile.lock` is not synchronized with your `Pipfile`. To fix this, run the following command in your project's root directory (where `Pipfile` is located) on your **host machine**:

```bash
pipenv lock
```

This will update `Pipfile.lock`. After running this command, proceed to Step 3.

### Step 3: Build and Run with Docker Compose

Open your terminal in the root directory of the project (where `docker-compose.yml` and `Dockerfile` are located).

**To build the image and run the application (e.g., execute `inference.py`):**

```bash
docker-compose up --build
```

- `--build`: This flag tells Docker Compose to build the Docker image using the `Dockerfile`. You can omit this on subsequent runs if the `Dockerfile` or its dependencies haven't changed, and an image already exists.
- The application (defined by `CMD` in the `Dockerfile`, e.g., `python3 inference.py`) will start, and its output will be displayed in your terminal.

**To run in detached mode (in the background):**

```bash
docker-compose up --build -d
```

### Step 4: Interacting with the Application

- **Viewing Logs (if running in detached mode):**

  ```bash
  docker-compose logs -f app
  ```

  (Replace `app` with your service name if it's different in `docker-compose.yml`). Press `Ctrl+C` to stop following logs.

- **Accessing a Shell Inside the Container (for debugging):**
  If you need to explore the container's environment or run commands manually:

  1.  Ensure the container is running (e.g., using `docker-compose up -d`).
  2.  Open a shell:
      ```bash
      docker-compose exec app bash
      ```
      (Replace `app` with your service name if it's different).
  3.  Inside the container, you can navigate to `/app` (the working directory) and run Python scripts or other commands.

- **Port Mapping (if applicable):**
  If your application (`inference.py`) runs a web server (e.g., on port 8000) and you have configured port mapping in `docker-compose.yml` (e.g., `ports: - "8000:8000"`), you can access it via `http://localhost:8000` in your web browser.

### Step 5: Stopping the Application

To stop and remove the containers, networks, and (optionally, depending on `docker-compose down` flags) volumes defined by Docker Compose:

```bash
docker-compose down
```

If you want to remove the volumes as well:

```bash
docker-compose down -v
```

## Important Notes

- **PyTorch Versions & CUDA:** The `Pipfile` specifies PyTorch versions and a CUDA source (`pytorch-cu111`). Ensure these versions are valid and available from the specified PyTorch wheel index. If `pipenv install` fails during the Docker build due to version conflicts or "Could not find a version" errors, you will need to:
  1.  Consult [PyTorch Previous Versions](https://pytorch.org/get-started/previous-versions/) to find compatible `torch`, `torchvision`, and `torchaudio` versions for your desired CUDA version (e.g., CUDA 11.1).
  2.  Update the versions in your `Pipfile`.
  3.  Run `pipenv lock` locally to regenerate `Pipfile.lock`.
  4.  Re-run `docker-compose up --build`.
- **GPU Usage:** The `docker-compose.yml` is configured to attempt GPU access using NVIDIA. This requires the prerequisites mentioned above (NVIDIA drivers and NVIDIA Container Toolkit on the host). If GPUs are not available or not configured correctly, PyTorch will typically fall back to CPU mode.
- **Development Mode Volume Mount:** The `docker-compose.yml` includes `volumes: - .:/app`. This mounts your local project directory into the container. Code changes made locally will be reflected inside the container, which is useful for development. For production, you might remove this volume mount to rely solely on the code baked into the image.

## Further Actions

- **Cleaning up Docker Resources:**
  - To remove unused Docker images: `docker image prune`
  - To remove unused Docker volumes: `docker volume prune`
  - To remove unused Docker networks: `docker network prune`
  - To remove all unused Docker resources (images, containers, volumes, networks): `docker system prune -a` (Use with caution!)
